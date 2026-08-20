use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::Serialize;
use serde_json::{Value, json};

use super::{ControlError, ControlErrorKind, EventHub};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskState {
    Queued,
    Running,
    Completed,
    Failed,
    Cancelled,
}

impl TaskState {
    fn terminal(self) -> bool {
        matches!(self, Self::Completed | Self::Failed | Self::Cancelled)
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct TaskSnapshot {
    pub task_id: String,
    pub label: String,
    pub state: TaskState,
    pub progress: Option<f64>,
    pub phase: String,
    pub result: Option<Value>,
    pub error: Option<Value>,
    pub created_at_unix_ms: u128,
    pub completed_at_unix_ms: Option<u128>,
    pub cancellation_supported: bool,
    pub owner_session_id: String,
}

#[derive(Debug)]
pub struct TaskRegistry {
    tasks: Mutex<HashMap<String, TaskSnapshot>>,
    event_hub: Arc<EventHub>,
}

impl TaskRegistry {
    pub fn shared(event_hub: Arc<EventHub>) -> Arc<Self> {
        Arc::new(Self {
            tasks: Mutex::new(HashMap::new()),
            event_hub,
        })
    }

    pub fn create(
        &self,
        label: impl Into<String>,
        owner_session_id: impl Into<String>,
        cancellation_supported: bool,
    ) -> Result<TaskSnapshot, ControlError> {
        let id = crate::control::discovery::random_uuid_like().map_err(|error| {
            ControlError::new(
                ControlErrorKind::Internal,
                format!("failed to allocate task ID: {error}"),
            )
        })?;
        let snapshot = TaskSnapshot {
            task_id: format!("task:{id}"),
            label: label.into(),
            state: TaskState::Queued,
            progress: Some(0.0),
            phase: "queued".into(),
            result: None,
            error: None,
            created_at_unix_ms: now_unix_ms(),
            completed_at_unix_ms: None,
            cancellation_supported,
            owner_session_id: owner_session_id.into(),
        };
        self.tasks
            .lock()
            .expect("task registry poisoned")
            .insert(snapshot.task_id.clone(), snapshot.clone());
        self.publish("tasks.created", &snapshot);
        Ok(snapshot)
    }

    pub fn get(&self, task_id: &str) -> Result<TaskSnapshot, ControlError> {
        self.tasks
            .lock()
            .expect("task registry poisoned")
            .get(task_id)
            .cloned()
            .ok_or_else(|| task_not_found(task_id))
    }

    pub fn list(&self, include_finished: bool) -> Vec<TaskSnapshot> {
        let mut tasks = self
            .tasks
            .lock()
            .expect("task registry poisoned")
            .values()
            .filter(|task| include_finished || !task.state.terminal())
            .cloned()
            .collect::<Vec<_>>();
        tasks.sort_by_key(|task| task.created_at_unix_ms);
        tasks
    }

    pub fn mark_running(&self, task_id: &str) -> Result<TaskSnapshot, ControlError> {
        self.update(
            task_id,
            |task| {
                if task.state == TaskState::Cancelled {
                    return;
                }
                task.state = TaskState::Running;
                task.phase = "running".into();
                task.progress = None;
            },
            "tasks.progress",
        )
    }

    pub fn complete(&self, task_id: &str, result: Value) -> Result<TaskSnapshot, ControlError> {
        self.update(
            task_id,
            |task| {
                if task.state == TaskState::Cancelled {
                    return;
                }
                task.state = TaskState::Completed;
                task.phase = "completed".into();
                task.progress = Some(1.0);
                task.result = Some(result);
                task.completed_at_unix_ms = Some(now_unix_ms());
            },
            "tasks.completed",
        )
    }

    pub fn fail(&self, task_id: &str, error: &ControlError) -> Result<TaskSnapshot, ControlError> {
        self.update(
            task_id,
            |task| {
                if task.state == TaskState::Cancelled {
                    return;
                }
                task.state = TaskState::Failed;
                task.phase = "failed".into();
                task.progress = None;
                task.error = Some(error.to_json_rpc_error());
                task.completed_at_unix_ms = Some(now_unix_ms());
            },
            "tasks.failed",
        )
    }

    pub fn cancel(&self, task_id: &str) -> Result<TaskSnapshot, ControlError> {
        self.update(
            task_id,
            |task| {
                if !task.state.terminal() && task.cancellation_supported {
                    task.state = TaskState::Cancelled;
                    task.phase = "cancelled".into();
                    task.progress = None;
                    task.completed_at_unix_ms = Some(now_unix_ms());
                }
            },
            "tasks.cancelled",
        )
        .and_then(|snapshot| {
            if !snapshot.cancellation_supported && !snapshot.state.terminal() {
                Err(ControlError::new(
                    ControlErrorKind::Unsupported,
                    "this task cannot be cancelled once submitted",
                )
                .with_data(json!({"task_id": task_id})))
            } else {
                Ok(snapshot)
            }
        })
    }

    pub fn progress(
        &self,
        task_id: &str,
        progress: Option<f64>,
        phase: impl Into<String>,
    ) -> Result<TaskSnapshot, ControlError> {
        let phase = phase.into();
        self.update(
            task_id,
            |task| {
                if task.state == TaskState::Running {
                    task.progress = progress.map(|value| value.clamp(0.0, 1.0));
                    task.phase = phase;
                }
            },
            "tasks.progress",
        )
    }

    pub fn forget(&self, task_id: &str) -> Result<(), ControlError> {
        let mut tasks = self.tasks.lock().expect("task registry poisoned");
        let task = tasks.get(task_id).ok_or_else(|| task_not_found(task_id))?;
        if !task.state.terminal() {
            return Err(ControlError::new(
                ControlErrorKind::Conflict,
                "a running task cannot be forgotten",
            )
            .with_data(json!({"task_id": task_id})));
        }
        tasks.remove(task_id);
        Ok(())
    }

    fn update(
        &self,
        task_id: &str,
        update: impl FnOnce(&mut TaskSnapshot),
        event: &str,
    ) -> Result<TaskSnapshot, ControlError> {
        let snapshot = {
            let mut tasks = self.tasks.lock().expect("task registry poisoned");
            let task = tasks
                .get_mut(task_id)
                .ok_or_else(|| task_not_found(task_id))?;
            update(task);
            task.clone()
        };
        self.publish(event, &snapshot);
        Ok(snapshot)
    }

    fn publish(&self, event: &str, snapshot: &TaskSnapshot) {
        self.event_hub.publish(
            event,
            snapshot.task_id.clone(),
            self.event_hub.revision(),
            serde_json::to_value(snapshot).unwrap_or_else(|_| json!({})),
            Some(snapshot.owner_session_id.clone()),
            None,
        );
    }
}

fn task_not_found(task_id: &str) -> ControlError {
    ControlError::new(
        ControlErrorKind::ResourceNotFound,
        format!("task '{task_id}' was not found"),
    )
    .with_data(json!({"task_id": task_id}))
}

fn now_unix_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tasks_transition_and_enforce_retention_rules() {
        let registry = TaskRegistry::shared(EventHub::shared());
        let task = registry
            .create("test operation", "session", true)
            .expect("create task");
        registry.mark_running(&task.task_id).expect("run task");
        assert!(registry.forget(&task.task_id).is_err());
        let task = registry
            .complete(&task.task_id, json!({"answer": 42}))
            .expect("complete task");
        assert_eq!(task.state, TaskState::Completed);
        registry.forget(&task.task_id).expect("forget task");
        assert!(registry.get(&task.task_id).is_err());
    }

    #[test]
    fn queued_tasks_can_be_cancelled() {
        let registry = TaskRegistry::shared(EventHub::shared());
        let task = registry.create("test", "session", true).expect("create");
        let task = registry.cancel(&task.task_id).expect("cancel");
        assert_eq!(task.state, TaskState::Cancelled);
    }
}
