//! Native renderer coverage contract for application-owned shell mounts.

pub(crate) const PROJECT_BUILTIN_MOUNTS: &[&str] = &[
    "builtin:project-top-bar",
    "builtin:project-workspace",
    "builtin:extension-host.top-bar-actions",
    "builtin:extension-host.status-bar",
    "builtin:extension-host.project-cards",
    "builtin:shell-inspector",
    "builtin:command-toolbar",
    "builtin:help",
    "builtin:recovery-controls",
];

pub(crate) const SINGLE_BUILTIN_MOUNTS: &[&str] = &[
    "builtin:viewer-top-bar",
    "builtin:viewer-canvas",
    "builtin:extension-host.top-bar-actions",
    "builtin:extension-host.status-bar",
    "builtin:extension-host.left-sections",
    "builtin:extension-host.right-tabs",
    "builtin:extension-host.canvas-controls",
    "builtin:shell-inspector",
    "builtin:command-toolbar",
    "builtin:help",
    "builtin:recovery-controls",
    "builtin:channels",
    "builtin:viewer-viewport-controls",
    "builtin:layers",
    "builtin:project",
    "builtin:properties",
    "builtin:views",
    "builtin:analysis",
    "builtin:measurements",
    "builtin:memory",
    "builtin:roi-selector",
];

pub(crate) const MOSAIC_BUILTIN_MOUNTS: &[&str] = &[
    "builtin:mosaic-top-bar",
    "builtin:mosaic-canvas",
    "builtin:extension-host.top-bar-actions",
    "builtin:extension-host.status-bar",
    "builtin:extension-host.left-sections",
    "builtin:extension-host.right-tabs",
    "builtin:extension-host.canvas-controls",
    "builtin:shell-inspector",
    "builtin:command-toolbar",
    "builtin:help",
    "builtin:recovery-controls",
    "builtin:channels",
    "builtin:layers",
    "builtin:project",
    "builtin:properties",
    "builtin:views",
    "builtin:mosaic-layout",
    "builtin:memory",
];

pub(crate) fn builtin_mounts(mode: &str) -> &'static [&'static str] {
    match mode {
        "project" => PROJECT_BUILTIN_MOUNTS,
        "single" => SINGLE_BUILTIN_MOUNTS,
        "mosaic" => MOSAIC_BUILTIN_MOUNTS,
        _ => &[],
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;

    #[test]
    fn component_catalogue_matches_every_native_renderer_dispatch_table() {
        for mode in ["project", "single", "mosaic"] {
            let catalog = odon::model::shell_component_catalog(mode);
            let advertised = catalog["components"]
                .as_array()
                .unwrap()
                .iter()
                .filter_map(|component| component["id"].as_str())
                .collect::<BTreeSet<_>>();
            let supported = builtin_mounts(mode)
                .iter()
                .copied()
                .collect::<BTreeSet<_>>();
            assert_eq!(advertised, supported, "{mode} catalogue/renderer drift");

            let renderer_source = match mode {
                "project" => include_str!("../root_app.rs").to_string(),
                "single" => format!(
                    "{}\n{}",
                    include_str!("../app/shell.rs"),
                    include_str!("../app/shell_components.rs")
                ),
                "mosaic" => format!(
                    "{}\n{}",
                    include_str!("../mosaic/shell.rs"),
                    include_str!("../mosaic/shell_components.rs")
                ),
                _ => unreachable!(),
            };
            for mount in supported {
                if mount.starts_with("builtin:extension-host.") {
                    assert!(
                        renderer_source.contains("mount.starts_with(\"builtin:extension-host.\")")
                    );
                } else {
                    assert!(
                        renderer_source.contains(&format!("\"{mount}\"")),
                        "{mode} advertises {mount} without a native dispatch branch"
                    );
                }
            }
        }
    }
}
