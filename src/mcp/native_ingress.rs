//! Native-UI command ingress: the renderer's non-blocking path into the control actor.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use crossbeam_channel::{Receiver, Sender};
use serde_json::{Value, json};

use crate::control::{
    ControlCommand, ControlError, ControlErrorKind, EventHub, OdonControlRequest, TaskRegistry,
};

const NATIVE_COMMAND_QUEUE_CAPACITY: usize = 512;
const NATIVE_REJECTION_LOG_CAPACITY: usize = 64;

/// Commands that set an absolute value and are emitted once per frame while a control is being
/// dragged. For these, and only these, a newer step makes an unsent older one redundant. Any
/// other command is always delivered, even when many of the same method are issued together.
const CONTINUOUS_SETTERS: &[&str] = &[
    "viewer.viewports.camera.set",
    "viewer.camera.set",
    "viewer.viewports.layers.state.replace",
    "viewer.viewports.channels.set_contrast",
    "viewer.channels.set_contrast",
    "viewer.channels.set_transform",
    "viewer.viewports.rendering.set",
    "viewer.viewports.objects.filter.set",
    "viewer.workspace.layout.set",
    "mosaic.layout.configure",
    "mosaic.objects.style.set",
    "mosaic.rendering.set",
];

/// Parameters that name what a setter targets; steps for different targets never coalesce.
const SETTER_TARGET_PARAMS: &[&str] = &[
    "viewport_id",
    "channel",
    "channels",
    "index",
    "layer_id",
    "target",
];

#[derive(Debug, Clone)]
pub struct NativeCommand {
    pub method: String,
    pub params: Value,
}

/// One native command waiting for, or travelling to, the control actor.
#[derive(Debug)]
struct QueuedNativeCommand {
    command: NativeCommand,
    /// Non-zero while the pointer gesture that produced the command is still in progress.
    gesture: u64,
}

impl QueuedNativeCommand {
    /// Identity of the continuous gesture this command is a step of, if any. While a pointer is
    /// held the user manipulates a single control, so consecutive steps of one absolute-value
    /// setter aimed at the same target are successive values of that control.
    fn gesture_key(&self) -> Option<String> {
        if self.gesture == 0 || !CONTINUOUS_SETTERS.contains(&self.command.method.as_str()) {
            return None;
        }
        let mut key = format!("{}:{}", self.gesture, self.command.method);
        for name in SETTER_TARGET_PARAMS {
            if let Some(value) = self.command.params.get(name) {
                key.push_str(&format!(":{name}={value}"));
            }
        }
        Some(key)
    }
}

#[derive(Debug, Default)]
struct NativeIngressShared {
    pending_methods: Mutex<HashMap<String, usize>>,
    /// Commands accepted by `submit` that the actor has not answered yet.
    outstanding: AtomicUsize,
    /// Generation of the pointer gesture in progress, or zero when no pointer is held.
    gesture: AtomicU64,
    gesture_serial: AtomicU64,
    rejections: Mutex<VecDeque<String>>,
}

impl NativeIngressShared {
    fn finish(&self, method: &str) {
        self.outstanding
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |value| {
                Some(value.saturating_sub(1))
            })
            .ok();
        clear_pending_method(&self.pending_methods, method);
    }

    fn reject(&self, method: &str, reason: &str) {
        let mut rejections = self
            .rejections
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if rejections.len() >= NATIVE_REJECTION_LOG_CAPACITY {
            rejections.pop_front();
        }
        rejections.push_back(format!("{method}: {reason}"));
    }
}

/// Native-UI ingress that forwards typed commands to the control actor without waiting for
/// another `RootApp::update` call.
///
/// Submission never blocks the UI thread. Commands are pipelined to the actor in submission
/// order, so a command whose reply is deferred (for example a load) does not delay the ones
/// behind it. Steps of one continuous pointer gesture are coalesced to a single command in flight
/// and are last-write-wins: a step that was built against a revision the gesture's own previous
/// step has since advanced is rebased instead of being dropped. Every other rejection is recorded
/// for the application log.
#[derive(Debug, Clone)]
pub struct NativeCommandIngress {
    tx: Sender<QueuedNativeCommand>,
    recorded_rx: Option<Receiver<QueuedNativeCommand>>,
    shared: Arc<NativeIngressShared>,
}

struct NativeCommandInFlight {
    queued: QueuedNativeCommand,
    key: Option<String>,
    reply: Receiver<Result<Value, ControlError>>,
    rebased: bool,
}

struct NativeIngressWorker {
    actor_tx: Sender<OdonControlRequest>,
    event_hub: Arc<EventHub>,
    task_registry: Arc<TaskRegistry>,
    shared: Arc<NativeIngressShared>,
    in_flight: Vec<NativeCommandInFlight>,
    /// Newest step of each gesture waiting for that gesture's in-flight command to be answered.
    held: Vec<(String, QueuedNativeCommand)>,
    /// Gestures that already had a step accepted by the actor.
    accepted_gestures: HashSet<String>,
}

impl NativeIngressWorker {
    fn run(mut self, rx: Receiver<QueuedNativeCommand>) {
        loop {
            let mut select = crossbeam_channel::Select::new();
            let command_index = select.recv(&rx);
            for in_flight in &self.in_flight {
                select.recv(&in_flight.reply);
            }
            let operation = select.select();
            let index = operation.index();
            if index == command_index {
                match operation.recv(&rx) {
                    Ok(queued) => {
                        if !self.accept(queued) {
                            break;
                        }
                    }
                    Err(_) => break,
                }
                continue;
            }
            let position = index - 1;
            let result = operation.recv(&self.in_flight[position].reply);
            let finished = self.in_flight.remove(position);
            if !self.complete(finished, result.ok()) {
                break;
            }
        }
    }

    /// Returns false once the actor mailbox is gone.
    fn accept(&mut self, queued: QueuedNativeCommand) -> bool {
        let key = queued.gesture_key();
        if let Some(key) = key.as_ref()
            && self
                .in_flight
                .iter()
                .any(|in_flight| in_flight.key.as_ref() == Some(key))
        {
            if let Some(slot) = self.held.iter_mut().find(|(held, _)| held == key) {
                let superseded = std::mem::replace(&mut slot.1, queued);
                self.shared.finish(&superseded.command.method);
            } else {
                self.held.push((key.clone(), queued));
            }
            return true;
        }
        // Anything else must not overtake a gesture step that was submitted before it.
        for (_, held) in std::mem::take(&mut self.held) {
            if !self.send(held, false) {
                return false;
            }
        }
        self.send(queued, false)
    }

    fn send(&mut self, queued: QueuedNativeCommand, rebased: bool) -> bool {
        let command = match ControlCommand::decode(
            &queued.command.method,
            queued.command.params.clone(),
        ) {
            Ok(command) => command,
            Err(error) => {
                self.shared
                    .reject(&queued.command.method, &error.message);
                self.shared.finish(&queued.command.method);
                return true;
            }
        };
        let (reply_tx, reply) = crossbeam_channel::bounded(1);
        if self
            .actor_tx
            .send(OdonControlRequest {
                command,
                reply: reply_tx,
                session_id: "native-ui".to_string(),
                request_id: None,
                event_hub: Arc::clone(&self.event_hub),
                task_registry: Arc::clone(&self.task_registry),
                task_id: None,
            })
            .is_err()
        {
            self.shared.finish(&queued.command.method);
            return false;
        }
        let key = queued.gesture_key();
        self.in_flight.push(NativeCommandInFlight {
            queued,
            key,
            reply,
            rebased,
        });
        true
    }

    fn complete(
        &mut self,
        finished: NativeCommandInFlight,
        result: Option<Result<Value, ControlError>>,
    ) -> bool {
        let NativeCommandInFlight {
            mut queued,
            key,
            rebased,
            ..
        } = finished;
        let newer = key.as_ref().and_then(|key| {
            self.held
                .iter()
                .position(|(held, _)| held == key)
                .map(|position| self.held.remove(position).1)
        });
        match result {
            Some(Ok(_)) => {
                if let Some(key) = key {
                    if self.accepted_gestures.len() >= 64 {
                        self.accepted_gestures.clear();
                    }
                    self.accepted_gestures.insert(key);
                }
            }
            Some(Err(error)) => {
                // A gesture step is built against the revision the renderer last saw. If only
                // the gesture's own earlier step advanced it, the newest value still wins.
                let rebase = (newer.is_none()
                    && !rebased
                    && error.kind == ControlErrorKind::Conflict
                    && key
                        .as_ref()
                        .is_some_and(|key| self.accepted_gestures.contains(key)))
                .then(|| revision_rebase(&error))
                .flatten();
                if let Some((field, revision)) = rebase
                    && queued.command.params.get(field).is_some()
                {
                    queued.command.params[field] = json!(revision);
                    return self.send(queued, true);
                }
                if newer.is_none() || error.kind != ControlErrorKind::Conflict {
                    self.shared.reject(&queued.command.method, &error.message);
                }
            }
            None => {}
        }
        self.shared.finish(&queued.command.method);
        match newer {
            Some(newer) => self.send(newer, false),
            None => true,
        }
    }
}

fn revision_rebase(error: &ControlError) -> Option<(&'static str, u64)> {
    let data = error.data.as_ref()?;
    let field = match data.get("revision_domain")?.as_str()? {
        "navigation" => "if_navigation_revision",
        "presentation" => "if_presentation_revision",
        _ => return None,
    };
    Some((field, data.get("current_revision")?.as_u64()?))
}

impl NativeCommandIngress {
    pub(super) fn actor(
        actor_tx: Sender<OdonControlRequest>,
        event_hub: Arc<EventHub>,
        task_registry: Arc<TaskRegistry>,
    ) -> anyhow::Result<Self> {
        let (tx, rx) = crossbeam_channel::unbounded::<QueuedNativeCommand>();
        let shared = Arc::new(NativeIngressShared::default());
        let worker = NativeIngressWorker {
            actor_tx,
            event_hub,
            task_registry,
            shared: Arc::clone(&shared),
            in_flight: Vec::new(),
            held: Vec::new(),
            accepted_gestures: HashSet::new(),
        };
        std::thread::Builder::new()
            .name("odon-native-command-ingress".to_string())
            .spawn(move || worker.run(rx))?;
        Ok(Self {
            tx,
            recorded_rx: None,
            shared,
        })
    }

    /// A bounded, non-consuming ingress used by renderer-only tests and before a renderer is
    /// attached to its required application runtime.
    pub fn detached() -> Self {
        let (tx, rx) = crossbeam_channel::unbounded::<QueuedNativeCommand>();
        Self {
            tx,
            recorded_rx: Some(rx),
            shared: Arc::new(NativeIngressShared::default()),
        }
    }

    /// Queues a command without blocking. Returns false, and records the rejection, when the
    /// ingress already holds its maximum number of unanswered commands.
    pub fn submit(&self, method: impl Into<String>, params: Value) -> bool {
        let method = method.into();
        if self
            .shared
            .outstanding
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |value| {
                (value < NATIVE_COMMAND_QUEUE_CAPACITY).then_some(value + 1)
            })
            .is_err()
        {
            self.shared
                .reject(&method, "native command queue is full; command dropped");
            return false;
        }
        *self
            .shared
            .pending_methods
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .entry(method.clone())
            .or_default() += 1;
        let queued = QueuedNativeCommand {
            command: NativeCommand {
                method: method.clone(),
                params,
            },
            gesture: self.shared.gesture.load(Ordering::Acquire),
        };
        if self.tx.send(queued).is_ok() {
            true
        } else {
            self.shared.finish(&method);
            false
        }
    }

    /// Reports whether a pointer button is held. Commands submitted while one is held are steps
    /// of a continuous gesture and are coalesced and rebased as described on the type.
    pub fn set_gesture_active(&self, active: bool) {
        let current = self.shared.gesture.load(Ordering::Acquire);
        if active && current == 0 {
            let next = self.shared.gesture_serial.fetch_add(1, Ordering::AcqRel) + 1;
            self.shared.gesture.store(next, Ordering::Release);
        } else if !active && current != 0 {
            self.shared.gesture.store(0, Ordering::Release);
        }
    }

    /// Drains the commands the actor rejected since the last call, formatted for the log.
    pub fn take_rejections(&self) -> Vec<String> {
        self.shared
            .rejections
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .drain(..)
            .collect()
    }

    pub fn contains_pending(&self, method: &str) -> bool {
        self.shared
            .pending_methods
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(method)
            .is_some_and(|count| *count > 0)
    }

    pub fn take_recorded(&self) -> Vec<NativeCommand> {
        let commands: Vec<NativeCommand> = self
            .recorded_rx
            .as_ref()
            .map(|rx| rx.try_iter().map(|queued| queued.command).collect())
            .unwrap_or_default();
        for command in &commands {
            self.shared.finish(&command.method);
        }
        commands
    }
}

fn clear_pending_method(pending_methods: &Mutex<HashMap<String, usize>>, method: &str) {
    let mut pending = pending_methods
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if let Some(count) = pending.get_mut(method) {
        *count = count.saturating_sub(1);
        if *count == 0 {
            pending.remove(method);
        }
    }
}

