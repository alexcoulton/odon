pub(crate) mod canvas_overlays;
pub(crate) mod channel_notes;
pub(crate) mod channels_panel;
pub(crate) mod command_palette;
pub(crate) mod command_shortcuts;
pub(crate) mod command_toolbar;
pub(crate) mod contrast;
pub(crate) mod group_layers;
pub(crate) mod help;
pub(crate) mod icons;
pub(crate) mod layer_list;
pub(crate) mod left_panel;
pub(crate) mod range_slider;
pub(crate) mod right_panel;
pub(crate) mod roi_browser;
pub(crate) mod shell_inspector;
pub(crate) mod shell_layout;
#[cfg(test)]
pub(crate) mod shell_mounts;
pub(crate) mod shell_recovery;
pub(crate) mod shell_tree;
pub(crate) mod style;
pub(crate) mod tooltip;
pub(crate) mod top_bar;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CommandPresentationInvocation {
    pub command_id: String,
    pub checked: Option<bool>,
}
