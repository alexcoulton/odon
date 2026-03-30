use std::any::type_name;
use std::collections::{HashMap, HashSet};

use eframe::egui;

use crate::data::project_config::{
    ProjectChannelGroup, ProjectChannelGroupMember, ProjectLayerGroups,
};
use crate::project::groups as layer_groups;
use crate::ui::icons::Icon;
use crate::ui::layer_list as ui_layer_list;

pub(crate) trait ChannelListHost {
    type LayerId: Copy + Eq + std::hash::Hash;

    fn channel_search(&self) -> &str;
    fn channel_search_mut(&mut self) -> &mut String;
    fn channel_count(&self) -> usize;
    fn channel_order(&self) -> &[usize];
    fn channel_name(&self, idx: usize) -> Option<String>;
    fn channel_visible(&self, idx: usize) -> Option<bool>;
    fn set_channel_visible(&mut self, idx: usize, visible: bool);
    fn channel_available(&self, idx: usize) -> bool;
    fn is_channel_selected(&self, idx: usize) -> bool;
    fn selected_channel_group_id(&self) -> Option<u64>;
    fn select_channel_group(&mut self, group_id: Option<u64>);
    fn handle_channel_primary_click(
        &mut self,
        idx: usize,
        visible_indices: &[usize],
        modifiers: egui::Modifiers,
    );
    fn handle_channel_secondary_click(&mut self, idx: usize);
    fn open_group_layers_dialog_channels(&mut self, members: Vec<usize>);
    fn layer_groups(&self) -> ProjectLayerGroups;
    fn set_layer_groups(&mut self, groups: ProjectLayerGroups);
    fn channels_changed(&mut self);
    fn layer_drag_mut(&mut self) -> &mut Option<ui_layer_list::LayerDragState<Self::LayerId>>;
    fn dragging_channel_idx(&self) -> Option<usize>;
    fn channel_layer_id(&self, idx: usize) -> Self::LayerId;
}

pub(crate) fn show<H: ChannelListHost>(host: &mut H, ui: &mut egui::Ui, ctx: &egui::Context) {
    let mut channels_changed = false;
    let mut groups_cfg = host.layer_groups();

    let mut chans_all = true;
    let mut chans_none = true;
    for idx in 0..host.channel_count() {
        if let Some(visible) = host.channel_visible(idx) {
            chans_all &= visible;
            chans_none &= !visible;
        }
    }
    let chans_mixed = !chans_all && !chans_none;

    let mut groups_changed = false;
    let mut create_group_clicked = false;
    let mut selection_changed = false;
    ui.horizontal(|ui| {
        ui.label(format!("Channels ({})", host.channel_count()));
        ui.add_space(4.0);
        let mut all = chans_all;
        if ui
            .add(egui::Checkbox::new(&mut all, "All").indeterminate(chans_mixed))
            .changed()
        {
            for idx in 0..host.channel_count() {
                host.set_channel_visible(idx, all);
            }
            channels_changed = true;
        }
        create_group_clicked |= ui.button("+ Group").clicked();
    });
    ui.horizontal(|ui| {
        ui.label("Search");
        let search = host.channel_search_mut();
        ui.add(
            egui::TextEdit::singleline(search)
                .hint_text("Filter channels...")
                .desired_width(180.0),
        );
        if !search.is_empty() && ui.button("Clear").clicked() {
            search.clear();
        }
    });

    if create_group_clicked {
        let existing = groups_cfg
            .channel_groups
            .iter()
            .map(|g| g.id)
            .collect::<Vec<_>>();
        let id = layer_groups::next_group_id(&existing);
        groups_cfg.channel_groups.push(ProjectChannelGroup {
            id,
            name: format!("Group {id}"),
            expanded: true,
            color_rgb: [255, 255, 255],
        });
        groups_changed = true;
    }

    let dragging_channel_idx = host.dragging_channel_idx();
    let mut drop_channel_to_group: Option<u64> = None;
    let mut drop_channel_to_ungroup = false;

    let channel_search = host.channel_search().trim().to_string();
    let ChannelPartition {
        visible_channel_indices,
        members_by_group,
        ungrouped,
    } = partition_channels_for_display(host, &groups_cfg, &channel_search);
    let ungrouped_range_indices = ungrouped
        .iter()
        .map(|&(_pos, ch_idx)| ch_idx)
        .collect::<Vec<_>>();

    if !groups_cfg.channel_groups.is_empty() {
        ui.add_space(6.0);
        ui.label("Groups");
    }

    let mut delete_group: Option<u64> = None;
    for group_idx in 0..groups_cfg.channel_groups.len() {
        let group_id = groups_cfg.channel_groups[group_idx].id;
        let members = members_by_group
            .get(&group_id)
            .map(|v| v.as_slice())
            .unwrap_or(&[]);
        let group_range_indices = members
            .iter()
            .map(|&(_pos, ch_idx)| ch_idx)
            .collect::<Vec<_>>();
        let mut all = true;
        let mut none = true;
        for &(_pos, ch_idx) in members {
            if let Some(visible) = host.channel_visible(ch_idx) {
                all &= visible;
                none &= !visible;
            }
        }
        let mixed = !members.is_empty() && !all && !none;

        let (mut group_name, mut group_expanded, mut group_color_rgb) = {
            let g = &groups_cfg.channel_groups[group_idx];
            (g.name.clone(), g.expanded, g.color_rgb)
        };

        let group_selected = host.selected_channel_group_id() == Some(group_id);
        let mut group_header_clicked = false;
        let header = egui::collapsing_header::CollapsingState::load_with_default_open(
            ui.ctx(),
            ui.make_persistent_id(("channel-group", type_name::<H>(), group_id)),
            group_expanded,
        )
        .show_header(ui, |ui| {
            let row_width = ui.available_width().max(1.0);
            let mut checkbox_rect: Option<egui::Rect> = None;
            let mut frame = egui::Frame::new()
                .inner_margin(egui::Margin::symmetric(6, 4))
                .corner_radius(ui.visuals().widgets.inactive.corner_radius)
                .begin(ui);

            frame.content_ui.allocate_ui_with_layout(
                egui::vec2(row_width, 24.0),
                egui::Layout::left_to_right(egui::Align::Center),
                |ui| {
                    let mut set_all = all;
                    ui.add_enabled_ui(!members.is_empty(), |ui| {
                        let vis_resp = ui
                            .add(egui::Checkbox::new(&mut set_all, "").indeterminate(mixed))
                            .on_hover_text("Toggle all channels in group");
                        checkbox_rect = Some(vis_resp.rect);
                        if vis_resp.changed() {
                            for &(_pos, ch_idx) in members {
                                host.set_channel_visible(ch_idx, set_all);
                            }
                            channels_changed = true;
                        }
                    });

                    let swatch_color = egui::Color32::from_rgb(
                        group_color_rgb[0],
                        group_color_rgb[1],
                        group_color_rgb[2],
                    );
                    let (swatch_rect, _) =
                        ui.allocate_exact_size(egui::vec2(12.0, 12.0), egui::Sense::hover());
                    ui.painter().rect_filled(
                        swatch_rect,
                        egui::CornerRadius::same(3),
                        swatch_color,
                    );

                    ui.add(egui::Label::new(group_name.clone()).selectable(false));
                    ui.label(
                        egui::RichText::new(format!("({})", members.len()))
                            .small()
                            .weak(),
                    );
                },
            );

            let row_rect = frame.frame.widget_rect(frame.content_ui.min_rect());
            let frame_resp = frame.allocate_space(ui);
            let row_hovered = ui
                .input(|i| i.pointer.hover_pos())
                .is_some_and(|p| row_rect.contains(p));
            if row_hovered
                && ctx.input(|i| i.pointer.button_clicked(egui::PointerButton::Primary))
                && ui
                    .input(|i| i.pointer.hover_pos())
                    .is_some_and(|p| !checkbox_rect.is_some_and(|rect| rect.contains(p)))
            {
                group_header_clicked = true;
            }
            if group_selected {
                frame.frame.fill = ui.visuals().selection.bg_fill;
                frame.frame.stroke = ui.visuals().selection.stroke;
            } else if row_hovered || frame_resp.hovered() {
                frame.frame.fill = ui.visuals().widgets.hovered.bg_fill;
                frame.frame.stroke = ui.visuals().widgets.hovered.bg_stroke;
            } else {
                frame.frame.fill = ui.visuals().widgets.inactive.bg_fill;
                frame.frame.stroke = ui.visuals().widgets.inactive.bg_stroke;
            }
            frame.paint(ui);
        });
        let open = header.is_open();
        if open != group_expanded {
            groups_changed = true;
        }
        let (_toggle, hdr, _body) = header.body(|ui| {
            ui.horizontal(|ui| {
                ui.label("Name");
                groups_changed |= ui.text_edit_singleline(&mut group_name).changed();
            });
            ui.horizontal(|ui| {
                ui.label("Color");
                let mut color = egui::Color32::from_rgb(
                    group_color_rgb[0],
                    group_color_rgb[1],
                    group_color_rgb[2],
                );
                if ui.color_edit_button_srgba(&mut color).changed() {
                    group_color_rgb = [color.r(), color.g(), color.b()];
                    groups_changed = true;
                    channels_changed = true;
                }
                if ui.button("Delete group").clicked() {
                    delete_group = Some(group_id);
                }
            });

            if members.is_empty() {
                ui.label("(empty)");
            } else {
                for &(pos, ch_idx) in members {
                    render_channel_row(
                        host,
                        ui,
                        ctx,
                        ch_idx,
                        pos,
                        &group_range_indices,
                        &visible_channel_indices,
                        &mut channels_changed,
                    );
                }
            }
        });
        if group_header_clicked || hdr.response.clicked() {
            host.select_channel_group(Some(group_id));
            selection_changed = true;
        }
        if let (Some(_dragging), Some(pointer)) =
            (dragging_channel_idx, ctx.input(|i| i.pointer.hover_pos()))
        {
            let rect = hdr.response.rect;
            if rect.contains(pointer) {
                drop_channel_to_group = Some(group_id);
                ui.painter().rect_stroke(
                    rect.expand(1.0),
                    egui::CornerRadius::same(4),
                    egui::Stroke::new(1.5, egui::Color32::from_rgb(120, 200, 255)),
                    egui::StrokeKind::Inside,
                );
            }
        }
        group_expanded = open;
        {
            let g = &mut groups_cfg.channel_groups[group_idx];
            g.name = group_name;
            g.expanded = group_expanded;
            g.color_rgb = group_color_rgb;
        }
    }

    if let Some(group_id) = delete_group {
        groups_cfg.channel_groups.retain(|g| g.id != group_id);
        groups_cfg
            .channel_members
            .retain(|_k, m| m.group_id != group_id);
        if host.selected_channel_group_id() == Some(group_id) {
            host.select_channel_group(None);
            selection_changed = true;
        }
        groups_changed = true;
    }

    if !ungrouped.is_empty() {
        ui.add_space(6.0);
        let ungrouped_header = ui.add(egui::Label::new("Ungrouped").sense(egui::Sense::click()));
        if ungrouped_header.clicked() && host.selected_channel_group_id().is_some() {
            host.select_channel_group(None);
            selection_changed = true;
        }
        if let (Some(_dragging), Some(pointer)) =
            (dragging_channel_idx, ctx.input(|i| i.pointer.hover_pos()))
        {
            if ungrouped_header.rect.contains(pointer) {
                drop_channel_to_ungroup = true;
                ui.painter().rect_stroke(
                    ungrouped_header.rect.expand(1.0),
                    egui::CornerRadius::same(4),
                    egui::Stroke::new(1.5, egui::Color32::from_rgb(120, 200, 255)),
                    egui::StrokeKind::Inside,
                );
            }
        }
        for &(pos, ch_idx) in &ungrouped {
            render_channel_row(
                host,
                ui,
                ctx,
                ch_idx,
                pos,
                &ungrouped_range_indices,
                &visible_channel_indices,
                &mut channels_changed,
            );
        }
    }

    if let Some(ch_idx) = dragging_channel_idx {
        if ctx.input(|i| i.pointer.button_released(egui::PointerButton::Primary))
            && (drop_channel_to_group.is_some() || drop_channel_to_ungroup)
        {
            if let Some(name) = host.channel_name(ch_idx) {
                if let Some(gid) = drop_channel_to_group {
                    groups_cfg.channel_members.insert(
                        name,
                        ProjectChannelGroupMember {
                            group_id: gid,
                            inherit_color: true,
                        },
                    );
                    groups_changed = true;
                } else if drop_channel_to_ungroup
                    && groups_cfg.channel_members.remove(name.as_str()).is_some()
                {
                    groups_changed = true;
                }
            }
            *host.layer_drag_mut() = None;
        }
    }

    if groups_changed {
        host.set_layer_groups(groups_cfg);
        channels_changed = true;
    }
    if channels_changed {
        host.channels_changed();
        ctx.request_repaint();
    } else if selection_changed {
        ctx.request_repaint();
    }
}

fn render_channel_row<H: ChannelListHost>(
    host: &mut H,
    ui: &mut egui::Ui,
    ctx: &egui::Context,
    ch_idx: usize,
    pos: usize,
    range_indices: &[usize],
    visible_indices: &[usize],
    channels_changed: &mut bool,
) {
    let id = host.channel_layer_id(ch_idx);
    let available = host.channel_available(ch_idx);
    let selected = host.is_channel_selected(ch_idx);
    let name = host
        .channel_name(ch_idx)
        .unwrap_or_else(|| format!("Channel {ch_idx}"));
    let visible = host.channel_visible(ch_idx);
    let resp = ui_layer_list::ui_layer_row(
        ui,
        ctx,
        host.layer_drag_mut(),
        ui_layer_list::LayerGroup::Channels,
        pos,
        id,
        &name,
        ui_layer_list::LayerRowOptions {
            available,
            selected,
            icon: Icon::Image,
            visible,
            color_rgb: None,
        },
    );
    let modifiers = ctx.input(|i| i.modifiers);
    if resp.selected_clicked {
        host.handle_channel_primary_click(ch_idx, range_indices, modifiers);
    } else if resp.row_response.secondary_clicked() {
        host.handle_channel_secondary_click(ch_idx);
    }
    if let Some(visible) = resp.visible_changed {
        host.set_channel_visible(ch_idx, visible);
        *channels_changed = true;
    }

    resp.row_response.context_menu(|ui| {
        let selected = visible_indices
            .iter()
            .copied()
            .filter(|idx| host.is_channel_selected(*idx))
            .collect::<Vec<_>>();
        let can_group = selected.len() >= 2;
        if ui
            .add_enabled(can_group, egui::Button::new("Group layers..."))
            .clicked()
        {
            host.open_group_layers_dialog_channels(selected);
            ui.close();
        }
    });
}

fn fuzzy_name_score_local(query: &str, candidate: &str) -> Option<i32> {
    let query = normalize_search_token(query);
    if query.is_empty() {
        return Some(0);
    }
    let candidate = normalize_search_token(candidate);
    if candidate.is_empty() {
        return None;
    }
    if let Some(pos) = candidate.find(&query) {
        return Some(10_000 - pos as i32);
    }

    let mut score = 0i32;
    let mut search_from = 0usize;
    let mut last_match = None;
    for qch in query.chars() {
        let Some(rel_pos) = candidate[search_from..].find(qch) else {
            return None;
        };
        let pos = search_from + rel_pos;
        score += 100;
        if let Some(prev) = last_match {
            let gap = pos.saturating_sub(prev + 1) as i32;
            score -= gap.min(32);
        }
        last_match = Some(pos);
        search_from = pos + qch.len_utf8();
    }
    Some(score - candidate.len() as i32)
}

fn normalize_search_token(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for ch in text.chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch.to_ascii_lowercase());
        }
    }
    out
}

struct ChannelPartition {
    visible_channel_indices: Vec<usize>,
    members_by_group: HashMap<u64, Vec<(usize, usize)>>,
    ungrouped: Vec<(usize, usize)>,
}

fn partition_channels_for_display<H: ChannelListHost>(
    host: &H,
    groups_cfg: &ProjectLayerGroups,
    channel_search: &str,
) -> ChannelPartition {
    let valid_group_ids = groups_cfg
        .channel_groups
        .iter()
        .map(|g| g.id)
        .collect::<HashSet<_>>();

    let mut visible_channel_indices: Vec<usize> = Vec::new();
    let mut members_by_group: HashMap<u64, Vec<(usize, usize)>> = HashMap::new();
    let mut ungrouped: Vec<(usize, usize)> = Vec::new();

    for (pos, ch_idx) in host.channel_order().iter().copied().enumerate() {
        let Some(name) = host.channel_name(ch_idx) else {
            continue;
        };
        let group_id = groups_cfg
            .channel_members
            .get(name.as_str())
            .map(|m| m.group_id)
            .filter(|gid| valid_group_ids.contains(gid));

        if let Some(gid) = group_id {
            visible_channel_indices.push(ch_idx);
            members_by_group.entry(gid).or_default().push((pos, ch_idx));
            continue;
        }

        if !channel_search.is_empty() && fuzzy_name_score_local(channel_search, &name).is_none() {
            continue;
        }

        visible_channel_indices.push(ch_idx);
        ungrouped.push((pos, ch_idx));
    }

    ChannelPartition {
        visible_channel_indices,
        members_by_group,
        ungrouped,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::project_config::{
        ProjectChannelGroup, ProjectChannelGroupMember, ProjectLayerGroups,
    };

    #[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
    struct TestLayerId(usize);

    struct TestHost {
        channel_search: String,
        channel_order: Vec<usize>,
        channel_names: Vec<Option<String>>,
        layer_drag: Option<ui_layer_list::LayerDragState<TestLayerId>>,
    }

    impl ChannelListHost for TestHost {
        type LayerId = TestLayerId;

        fn channel_search(&self) -> &str {
            &self.channel_search
        }

        fn channel_search_mut(&mut self) -> &mut String {
            &mut self.channel_search
        }

        fn channel_count(&self) -> usize {
            self.channel_names.len()
        }

        fn channel_order(&self) -> &[usize] {
            &self.channel_order
        }

        fn channel_name(&self, idx: usize) -> Option<String> {
            self.channel_names.get(idx).cloned().flatten()
        }

        fn channel_visible(&self, _idx: usize) -> Option<bool> {
            Some(true)
        }

        fn set_channel_visible(&mut self, _idx: usize, _visible: bool) {}

        fn channel_available(&self, _idx: usize) -> bool {
            true
        }

        fn is_channel_selected(&self, _idx: usize) -> bool {
            false
        }

        fn selected_channel_group_id(&self) -> Option<u64> {
            None
        }

        fn select_channel_group(&mut self, _group_id: Option<u64>) {}

        fn handle_channel_primary_click(
            &mut self,
            _idx: usize,
            _visible_indices: &[usize],
            _modifiers: egui::Modifiers,
        ) {
        }

        fn handle_channel_secondary_click(&mut self, _idx: usize) {}

        fn open_group_layers_dialog_channels(&mut self, _members: Vec<usize>) {}

        fn layer_groups(&self) -> ProjectLayerGroups {
            ProjectLayerGroups::default()
        }

        fn set_layer_groups(&mut self, _groups: ProjectLayerGroups) {}

        fn channels_changed(&mut self) {}

        fn layer_drag_mut(&mut self) -> &mut Option<ui_layer_list::LayerDragState<Self::LayerId>> {
            &mut self.layer_drag
        }

        fn dragging_channel_idx(&self) -> Option<usize> {
            None
        }

        fn channel_layer_id(&self, idx: usize) -> Self::LayerId {
            TestLayerId(idx)
        }
    }

    #[test]
    fn search_only_filters_ungrouped_channels() {
        let host = TestHost {
            channel_search: "beta".to_string(),
            channel_order: vec![0, 1, 2, 3],
            channel_names: vec![
                Some("Alpha".to_string()),
                Some("Grouped One".to_string()),
                Some("Grouped Two".to_string()),
                Some("Beta".to_string()),
            ],
            layer_drag: None,
        };
        let groups_cfg = ProjectLayerGroups {
            channel_groups: vec![ProjectChannelGroup {
                id: 7,
                name: "Markers".to_string(),
                expanded: true,
                color_rgb: [255, 255, 255],
            }],
            channel_members: HashMap::from([
                (
                    "Grouped One".to_string(),
                    ProjectChannelGroupMember {
                        group_id: 7,
                        inherit_color: true,
                    },
                ),
                (
                    "Grouped Two".to_string(),
                    ProjectChannelGroupMember {
                        group_id: 7,
                        inherit_color: true,
                    },
                ),
            ]),
            ..ProjectLayerGroups::default()
        };

        let partition =
            partition_channels_for_display(&host, &groups_cfg, host.channel_search().trim());

        assert_eq!(partition.visible_channel_indices, vec![1, 2, 3]);
        assert_eq!(partition.ungrouped, vec![(3, 3)]);
        assert_eq!(
            partition.members_by_group.get(&7),
            Some(&vec![(1, 1), (2, 2)])
        );
    }

    #[test]
    fn invalid_group_membership_falls_back_to_filtered_ungrouped_list() {
        let host = TestHost {
            channel_search: "beta".to_string(),
            channel_order: vec![0, 1],
            channel_names: vec![Some("Alpha".to_string()), Some("Beta".to_string())],
            layer_drag: None,
        };
        let groups_cfg = ProjectLayerGroups {
            channel_members: HashMap::from([(
                "Alpha".to_string(),
                ProjectChannelGroupMember {
                    group_id: 99,
                    inherit_color: true,
                },
            )]),
            ..ProjectLayerGroups::default()
        };

        let partition =
            partition_channels_for_display(&host, &groups_cfg, host.channel_search().trim());

        assert!(partition.members_by_group.is_empty());
        assert_eq!(partition.visible_channel_indices, vec![1]);
        assert_eq!(partition.ungrouped, vec![(1, 1)]);
    }
}
