//! Application-shell method descriptors.

use super::*;

pub(super) fn methods() -> Vec<MethodDescriptor> {
    vec![
        method!(
            "ui.commands.describe_schema",
            "Describe command descriptors, shortcuts, platform-menu nodes, limits, and protection rules.",
            "ui.shell.read",
            false,
            false,
            None,
            ALL_MODES,
            Empty
        ),
        method!(
            "ui.commands.list",
            "List actor-owned application command descriptors independently of their presentations.",
            "ui.shell.read",
            false,
            false,
            None,
            ALL_MODES,
            Empty
        ),
        method!(
            "ui.commands.register",
            "Register or update an extension-owned command with a namespaced ID and conflict-checked shortcut.",
            "ui.shell.shortcuts",
            true,
            false,
            Some("ui.commands.changed"),
            ALL_MODES,
            CommandRegister
        ),
        method!(
            "ui.commands.remove",
            "Remove an owned extension command and every platform-menu presentation that references it.",
            "ui.shell.shortcuts",
            true,
            false,
            Some("ui.commands.changed"),
            ALL_MODES,
            CommandRemove
        ),
        method!(
            "ui.commands.execute",
            "Invoke any ready command through its actor-resolved native, control, or extension-event handler.",
            "ui.shell.read",
            false,
            false,
            Some("ui.commands.executed"),
            ALL_MODES,
            CommandExecute
        ),
        method!(
            "ui.commands.cleanup_extensions",
            "Reconcile extension-owned commands when their native connection closes.",
            "ui.shell.application_control",
            true,
            false,
            Some("ui.commands.changed"),
            ALL_MODES,
            CommandCleanup
        ),
        method!(
            "ui.commands.sync_extension",
            "Synchronize retained extension-command ownership, version compatibility, and readiness.",
            "ui.shell.application_control",
            true,
            false,
            Some("ui.commands.changed"),
            ALL_MODES,
            CommandSync
        ),
        method!(
            "ui.menus.get",
            "Inspect the revisioned declarative platform application menu.",
            "ui.shell.read",
            false,
            false,
            None,
            ALL_MODES,
            Empty
        ),
        method!(
            "ui.menus.replace",
            "Atomically replace the platform application-menu presentation while preserving protected commands.",
            "ui.shell.chrome",
            true,
            false,
            Some("ui.menus.changed"),
            ALL_MODES,
            MenuReplace
        ),
        method!(
            "ui.toolbars.get",
            "Inspect the revisioned declarative application command toolbar.",
            "ui.shell.read",
            false,
            false,
            None,
            ALL_MODES,
            Empty
        ),
        method!(
            "ui.toolbars.replace",
            "Atomically replace toolbar groups and command presentations.",
            "ui.shell.chrome",
            true,
            false,
            Some("ui.toolbars.changed"),
            ALL_MODES,
            ToolbarReplace
        ),
        method!(
            "ui.palette.get",
            "Inspect the revisioned searchable command-palette presentation.",
            "ui.shell.read",
            false,
            false,
            None,
            ALL_MODES,
            Empty
        ),
        method!(
            "ui.palette.replace",
            "Atomically replace command-palette title, prompt, shortcut, description visibility, and result limit.",
            "ui.shell.chrome",
            true,
            false,
            Some("ui.palette.changed"),
            ALL_MODES,
            PaletteReplace
        ),
        method!(
            "ui.shell.describe_schema",
            "Describe the versioned shell snapshot, patch, persistence, recovery, and event contracts.",
            "ui.shell.read",
            false,
            false,
            None,
            ALL_MODES,
            Empty
        ),
        method!(
            "ui.shell.components.list",
            "List built-in GUI component mounts and their composition constraints.",
            "ui.shell.read",
            false,
            false,
            None,
            ALL_MODES,
            ShellGet
        ),
        method!(
            "ui.shell.get",
            "Inspect the actor-owned application-shell tree and stable built-in node IDs.",
            "ui.shell.read",
            false,
            false,
            None,
            ALL_MODES,
            ShellGet
        ),
        method!(
            "ui.shell.export_layout",
            "Export one mode as a portable versioned shell-layout document.",
            "ui.shell.persistence",
            false,
            false,
            None,
            ALL_MODES,
            ShellGet
        ),
        method!(
            "ui.shell.import_layout",
            "Atomically validate, migrate, and import a portable shell-layout document.",
            "ui.shell.compose",
            true,
            false,
            Some("ui.shell.changed"),
            ALL_MODES,
            ShellImportLayout
        ),
        method!(
            "ui.shell.patch",
            "Atomically change shell visibility, child order, and selected built-in tabs.",
            "ui.shell.compose",
            true,
            false,
            Some("ui.shell.changed"),
            ALL_MODES,
            ShellPatch
        ),
        method!(
            "ui.shell.patch_layout",
            "Atomically update desired-tree visibility, selection, sizing, split, collapse, configuration, active-region, or focus state.",
            "ui.shell.compose",
            true,
            false,
            Some("ui.shell.changed"),
            ALL_MODES,
            ShellPatchLayout
        ),
        method!(
            "ui.shell.profiles.list",
            "List named shell layouts in session, application, or project scope.",
            "ui.shell.persistence",
            false,
            false,
            None,
            ALL_MODES,
            ShellProfileList
        ),
        method!(
            "ui.shell.profiles.load",
            "Atomically load a named shell layout into the active mode.",
            "ui.shell.persistence",
            true,
            false,
            Some("ui.shell.changed"),
            ALL_MODES,
            ShellProfileLoad
        ),
        method!(
            "ui.shell.profiles.remove",
            "Remove a named session, application, or project shell layout.",
            "ui.shell.persistence",
            true,
            false,
            Some("ui.shell.profiles.changed"),
            ALL_MODES,
            ShellProfileRemove
        ),
        method!(
            "ui.shell.profiles.save",
            "Save one mode's current shell layout under a session, application, or project name.",
            "ui.shell.persistence",
            true,
            false,
            Some("ui.shell.profiles.changed"),
            ALL_MODES,
            ShellProfileSave
        ),
        method!(
            "ui.shell.reset",
            "Reset one application mode to Odon's built-in shell layout.",
            "ui.shell.compose",
            true,
            false,
            Some("ui.shell.changed"),
            ALL_MODES,
            ShellReset
        ),
        method!(
            "ui.shell.recover",
            "Replace the active shell with Odon's protected minimal recovery layout.",
            "ui.shell.recovery",
            true,
            false,
            Some("ui.shell.changed"),
            ALL_MODES,
            ShellReset
        ),
        method!(
            "ui.shell.replace_layout",
            "Atomically replace the active mode's validated keyed application layout tree.",
            "ui.shell.compose",
            true,
            false,
            Some("ui.shell.changed"),
            ALL_MODES,
            ShellReplaceLayout
        ),
    ]
}
