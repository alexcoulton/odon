use std::collections::BTreeMap;

use serde_json::{Value, json};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum OperationKind {
    Document,
    DatasetInspection,
    RemoteListing,
    DeepLinkResolve,
    ProjectIo,
    Labels,
    Objects,
    ObjectFilter,
    MaskIo,
    SettingsIo,
}

impl OperationKind {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Document => "document",
            Self::DatasetInspection => "dataset_inspection",
            Self::RemoteListing => "remote_listing",
            Self::DeepLinkResolve => "deep_link_resolve",
            Self::ProjectIo => "project_io",
            Self::Labels => "labels",
            Self::Objects => "objects",
            Self::ObjectFilter => "object_filter",
            Self::MaskIo => "mask_io",
            Self::SettingsIo => "settings_io",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct OperationKey {
    kind: OperationKind,
    scope: Option<String>,
}

impl OperationKey {
    const fn unscoped(kind: OperationKind) -> Self {
        Self { kind, scope: None }
    }

    fn scoped(kind: OperationKind, scope: impl Into<String>) -> Self {
        Self {
            kind,
            scope: Some(scope.into()),
        }
    }

    fn snapshot_key(&self) -> String {
        match &self.scope {
            Some(scope) => format!("{}:{scope}", self.kind.as_str()),
            None => self.kind.as_str().to_string(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum OperationPhase {
    Pending,
    Ready,
    Failed,
    Cancelled,
}

impl OperationPhase {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::Ready => "ready",
            Self::Failed => "failed",
            Self::Cancelled => "cancelled",
        }
    }
}

#[derive(Debug, Clone)]
struct OperationState {
    generation: u64,
    phase: OperationPhase,
    status: String,
    sequence: u64,
}

#[derive(Debug, Clone)]
pub(crate) struct ReadinessModel {
    operations: BTreeMap<OperationKey, OperationState>,
    next_sequence: u64,
}

impl Default for ReadinessModel {
    fn default() -> Self {
        Self {
            operations: BTreeMap::new(),
            next_sequence: 1,
        }
    }
}

impl ReadinessModel {
    pub(crate) fn begin(
        &mut self,
        kind: OperationKind,
        generation: u64,
        status: impl Into<String>,
    ) {
        self.set(
            OperationKey::unscoped(kind),
            generation,
            OperationPhase::Pending,
            status,
        );
    }

    pub(crate) fn begin_scoped(
        &mut self,
        kind: OperationKind,
        scope: impl Into<String>,
        generation: u64,
        status: impl Into<String>,
    ) {
        self.set(
            OperationKey::scoped(kind, scope),
            generation,
            OperationPhase::Pending,
            status,
        );
    }

    pub(crate) fn finish(
        &mut self,
        kind: OperationKind,
        generation: u64,
        status: impl Into<String>,
    ) -> bool {
        self.set_if_current(
            &OperationKey::unscoped(kind),
            generation,
            OperationPhase::Ready,
            status,
        )
    }

    pub(crate) fn finish_scoped(
        &mut self,
        kind: OperationKind,
        scope: &str,
        generation: u64,
        status: impl Into<String>,
    ) -> bool {
        self.set_if_current(
            &OperationKey::scoped(kind, scope),
            generation,
            OperationPhase::Ready,
            status,
        )
    }

    pub(crate) fn fail(
        &mut self,
        kind: OperationKind,
        generation: u64,
        status: impl Into<String>,
    ) -> bool {
        self.set_if_current(
            &OperationKey::unscoped(kind),
            generation,
            OperationPhase::Failed,
            status,
        )
    }

    pub(crate) fn fail_scoped(
        &mut self,
        kind: OperationKind,
        scope: &str,
        generation: u64,
        status: impl Into<String>,
    ) -> bool {
        self.set_if_current(
            &OperationKey::scoped(kind, scope),
            generation,
            OperationPhase::Failed,
            status,
        )
    }

    pub(crate) fn cancel(
        &mut self,
        kind: OperationKind,
        generation: u64,
        status: impl Into<String>,
    ) -> bool {
        self.set_if_current(
            &OperationKey::unscoped(kind),
            generation,
            OperationPhase::Cancelled,
            status,
        )
    }

    pub(crate) fn cancel_scoped(
        &mut self,
        kind: OperationKind,
        scope: &str,
        generation: u64,
        status: impl Into<String>,
    ) -> bool {
        self.set_if_current(
            &OperationKey::scoped(kind, scope),
            generation,
            OperationPhase::Cancelled,
            status,
        )
    }

    pub(crate) fn mark_ready(
        &mut self,
        kind: OperationKind,
        generation: u64,
        status: impl Into<String>,
    ) {
        self.set(
            OperationKey::unscoped(kind),
            generation,
            OperationPhase::Ready,
            status,
        );
    }

    pub(crate) fn is_pending(&self, kind: OperationKind, generation: u64) -> bool {
        self.operations
            .get(&OperationKey::unscoped(kind))
            .is_some_and(|operation| {
                operation.generation == generation && operation.phase == OperationPhase::Pending
            })
    }

    pub(crate) fn is_pending_scoped(
        &self,
        kind: OperationKind,
        scope: &str,
        generation: u64,
    ) -> bool {
        self.operations
            .get(&OperationKey::scoped(kind, scope))
            .is_some_and(|operation| {
                operation.generation == generation && operation.phase == OperationPhase::Pending
            })
    }

    pub(crate) fn any_pending(&self) -> bool {
        self.operations
            .values()
            .any(|operation| operation.phase == OperationPhase::Pending)
    }

    pub(crate) fn status_for(&self, kind: OperationKind) -> Option<&str> {
        self.operations
            .iter()
            .filter(|(key, _)| key.kind == kind)
            .max_by_key(|(_, operation)| operation.sequence)
            .map(|(_, operation)| operation.status.as_str())
    }

    pub(crate) fn cancel_kind_pending(&mut self, kind: OperationKind, status: &str) {
        let pending = self
            .operations
            .iter()
            .filter_map(|(key, operation)| {
                (key.kind == kind && operation.phase == OperationPhase::Pending)
                    .then_some((key.clone(), operation.generation))
            })
            .collect::<Vec<_>>();
        for (key, generation) in pending {
            self.set_if_current(
                &key,
                generation,
                OperationPhase::Cancelled,
                status.to_string(),
            );
        }
    }

    pub(crate) fn aggregate_status(&self) -> &str {
        self.operations
            .values()
            .filter(|operation| operation.phase == OperationPhase::Pending)
            .max_by_key(|operation| operation.sequence)
            .or_else(|| {
                self.operations
                    .values()
                    .max_by_key(|operation| operation.sequence)
            })
            .map_or("Ready", |operation| operation.status.as_str())
    }

    pub(crate) fn cancel_all_pending(&mut self, status: &str) {
        let pending = self
            .operations
            .iter()
            .filter_map(|(key, operation)| {
                (operation.phase == OperationPhase::Pending)
                    .then_some((key.clone(), operation.generation))
            })
            .collect::<Vec<_>>();
        for (key, generation) in pending {
            self.set_if_current(
                &key,
                generation,
                OperationPhase::Cancelled,
                status.to_string(),
            );
        }
    }

    pub(crate) fn snapshot(&self) -> Value {
        let operations = self
            .operations
            .iter()
            .map(|(key, operation)| {
                (
                    key.snapshot_key(),
                    json!({
                        "kind":key.kind.as_str(),
                        "scope":key.scope,
                        "generation":operation.generation,
                        "phase":operation.phase.as_str(),
                        "busy":operation.phase == OperationPhase::Pending,
                        "ready":operation.phase == OperationPhase::Ready,
                        "status":operation.status,
                    }),
                )
            })
            .collect::<serde_json::Map<_, _>>();
        Value::Object(operations)
    }

    fn set_if_current(
        &mut self,
        key: &OperationKey,
        generation: u64,
        phase: OperationPhase,
        status: impl Into<String>,
    ) -> bool {
        if !self
            .operations
            .get(key)
            .is_some_and(|operation| operation.generation == generation)
        {
            return false;
        }
        self.set(key.clone(), generation, phase, status);
        true
    }

    fn set(
        &mut self,
        key: OperationKey,
        generation: u64,
        phase: OperationPhase,
        status: impl Into<String>,
    ) {
        let sequence = self.next_sequence;
        self.next_sequence = self.next_sequence.wrapping_add(1).max(1);
        self.operations.insert(
            key,
            OperationState {
                generation,
                phase,
                status: status.into(),
                sequence,
            },
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn independent_operations_do_not_clear_each_others_busy_state() {
        let mut readiness = ReadinessModel::default();
        readiness.begin(OperationKind::ProjectIo, 1, "Saving project");
        readiness.begin(OperationKind::Labels, 4, "Loading labels");

        assert!(readiness.finish(OperationKind::Labels, 4, "Labels ready"));
        assert!(readiness.any_pending());
        assert_eq!(readiness.aggregate_status(), "Saving project");
        assert_eq!(readiness.snapshot()["project_io"]["phase"], "pending");
        assert_eq!(readiness.snapshot()["labels"]["phase"], "ready");

        assert!(readiness.finish(OperationKind::ProjectIo, 1, "Ready"));
        assert!(!readiness.any_pending());
    }

    #[test]
    fn stale_completion_cannot_replace_a_newer_generation() {
        let mut readiness = ReadinessModel::default();
        readiness.begin(OperationKind::Objects, 1, "First load");
        readiness.begin(OperationKind::Objects, 2, "Second load");

        assert!(!readiness.finish(OperationKind::Objects, 1, "Stale ready"));
        assert!(readiness.is_pending(OperationKind::Objects, 2));
        assert_eq!(readiness.aggregate_status(), "Second load");
    }

    #[test]
    fn scoped_operations_of_the_same_kind_are_independent() {
        let mut readiness = ReadinessModel::default();
        readiness.begin_scoped(OperationKind::ObjectFilter, "left", 1, "Filtering left");
        readiness.begin_scoped(OperationKind::ObjectFilter, "right", 2, "Filtering right");

        assert!(readiness.finish_scoped(OperationKind::ObjectFilter, "left", 1, "Left ready"));
        assert!(readiness.any_pending());
        assert!(readiness.is_pending_scoped(OperationKind::ObjectFilter, "right", 2));
        assert_eq!(readiness.snapshot()["object_filter:left"]["phase"], "ready");
        assert_eq!(
            readiness.snapshot()["object_filter:right"]["phase"],
            "pending"
        );

        readiness.cancel_kind_pending(OperationKind::ObjectFilter, "Filters superseded");
        assert!(!readiness.any_pending());
        assert_eq!(
            readiness.snapshot()["object_filter:right"]["phase"],
            "cancelled"
        );
    }
}
