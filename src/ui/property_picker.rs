//! Searchable object-property picker with project-saved pins, recents, and automatic groups.

use std::collections::{BTreeMap, HashMap};

use eframe::egui;
use serde_json::{Value, json};

use crate::data::project_config::ObjectPropertyPreferences;

/// Keeps "recent" useful at a glance rather than becoming a second full list.
const MAX_RECENT_PROPERTIES: usize = 6;
/// A shared prefix only becomes a group once it is more than a naming coincidence.
const MIN_PREFIX_GROUP_SIZE: usize = 3;
/// How many frames to trust a commit over the project snapshot while it is applied.
const COMMIT_WAIT_SYNCS: u32 = 120;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum PropertyGroup {
    General,
    Markers,
    Prefixed(String),
    /// Coordinates, paths, and identifiers: rarely useful for colouring, so collapsed by default.
    Other,
}

impl PropertyGroup {
    fn title(&self) -> String {
        match self {
            Self::General => "General".to_string(),
            Self::Markers => "Markers".to_string(),
            Self::Prefixed(prefix) => format!("{prefix}_*"),
            Self::Other => "Other".to_string(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PropertyItem {
    pub id: String,
    pub label: String,
    pub group: PropertyGroup,
}

fn is_rarely_coloured(name: &str) -> bool {
    const PREFIXES: [&str; 6] = [
        "bbox_",
        "image_scale_",
        "instance_",
        "polygon_",
        "summary_",
        "x_",
    ];
    const SUFFIXES: [&str; 5] = ["_path", "_id", "_centroid", "_centroid_image", "_image"];
    matches!(name, "label" | "name" | "geometry" | "id")
        || name.starts_with("y_")
        || PREFIXES.iter().any(|prefix| name.starts_with(prefix))
        || SUFFIXES.iter().any(|suffix| name.ends_with(suffix))
}

/// Splits per-marker columns into a marker and a statistic, working back from the end because
/// marker names contain underscores: `marker_B7_H3_C_1_p90_intensity_log1p` is `B7_H3_C_1` and
/// `p90_log1p`, and `median_raw__cd3` is `cd3` and `median_raw`.
pub(crate) fn parse_marker_column(name: &str) -> Option<(String, String)> {
    if let Some(rest) = name.strip_prefix("marker_") {
        let (rest, transform) = match rest.strip_suffix("_log1p") {
            Some(rest) => (rest, "_log1p"),
            None => (rest, ""),
        };
        let rest = rest.strip_suffix("_intensity")?;
        let (marker, statistic) = rest.rsplit_once('_')?;
        return (!marker.is_empty() && !statistic.is_empty())
            .then(|| (marker.to_string(), format!("{statistic}{transform}")));
    }
    let (statistic, marker) = name.split_once("__")?;
    (!statistic.is_empty() && !marker.is_empty())
        .then(|| (marker.to_string(), statistic.to_string()))
}

fn prefix_of(name: &str) -> Option<&str> {
    name.split_once('_')
        .map(|(prefix, _)| prefix)
        .filter(|prefix| !prefix.is_empty())
}

/// Groups come from naming patterns, so any dataset gets them without configuration.
pub(crate) fn classify_properties(names: &[String]) -> Vec<PropertyItem> {
    let mut prefix_counts = HashMap::<&str, usize>::new();
    for name in names {
        if !is_rarely_coloured(name)
            && parse_marker_column(name).is_none()
            && let Some(prefix) = prefix_of(name)
        {
            *prefix_counts.entry(prefix).or_default() += 1;
        }
    }
    names
        .iter()
        .map(|name| {
            let group = if is_rarely_coloured(name) {
                PropertyGroup::Other
            } else if parse_marker_column(name).is_some() {
                PropertyGroup::Markers
            } else {
                match prefix_of(name) {
                    Some(prefix)
                        if prefix_counts.get(prefix).copied().unwrap_or(0)
                            >= MIN_PREFIX_GROUP_SIZE =>
                    {
                        PropertyGroup::Prefixed(prefix.to_string())
                    }
                    _ => PropertyGroup::General,
                }
            };
            PropertyItem {
                id: name.clone(),
                label: name.clone(),
                group,
            }
        })
        .collect()
}

/// Numeric columns seen as markers and statistics, so switching marker can keep the statistic.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct MarkerColumns {
    columns: BTreeMap<String, BTreeMap<String, String>>,
    /// Numeric columns that are not per-marker statistics.
    unparsed: Vec<String>,
}

impl MarkerColumns {
    pub(crate) fn from_columns(names: &[String]) -> Self {
        let mut parsed = Self::default();
        for name in names {
            match parse_marker_column(name) {
                Some((marker, statistic)) => {
                    parsed
                        .columns
                        .entry(marker)
                        .or_default()
                        .insert(statistic, name.clone());
                }
                None => parsed.unparsed.push(name.clone()),
            }
        }
        parsed
    }

    /// One marker has nothing to compare against, so it is not worth a second control.
    pub(crate) fn is_useful(&self) -> bool {
        self.columns.len() >= 2
    }

    pub(crate) fn items(&self) -> Vec<PropertyItem> {
        let mut items = self
            .columns
            .keys()
            .map(|marker| PropertyItem {
                id: marker.clone(),
                label: marker.clone(),
                group: PropertyGroup::Markers,
            })
            .collect::<Vec<_>>();
        items.extend(classify_properties(&self.unparsed));
        items
    }

    /// The picker entry and statistic for a column. Unparsed columns are their own entry.
    pub(crate) fn locate(&self, column: &str) -> (String, Option<String>) {
        match parse_marker_column(column) {
            Some((marker, statistic))
                if self
                    .columns
                    .get(&marker)
                    .is_some_and(|statistics| statistics.contains_key(&statistic)) =>
            {
                (marker, Some(statistic))
            }
            _ => (column.to_string(), None),
        }
    }

    pub(crate) fn statistics(&self, marker: &str) -> Vec<String> {
        self.columns
            .get(marker)
            .map(|statistics| statistics.keys().cloned().collect())
            .unwrap_or_default()
    }

    /// Keeps the statistic when this marker has it, otherwise falls back to its first.
    pub(crate) fn column_for(&self, entry: &str, statistic: Option<&str>) -> Option<String> {
        let Some(statistics) = self.columns.get(entry) else {
            return self
                .unparsed
                .iter()
                .find(|column| column.as_str() == entry)
                .cloned();
        };
        statistic
            .and_then(|statistic| statistics.get(statistic))
            .or_else(|| statistics.values().next())
            .cloned()
    }
}

pub(crate) fn statistic_label(statistic: &str) -> String {
    match statistic.strip_suffix("_log1p") {
        Some(base) => format!("{base} (log1p)"),
        None => statistic.to_string(),
    }
}

/// The project's pins and recents, plus edits on their way to the project model.
#[derive(Debug, Clone, Default)]
pub(crate) struct PropertyPreferencesDraft {
    preferences: ObjectPropertyPreferences,
    uncommitted: bool,
    awaiting: Option<(ObjectPropertyPreferences, u32)>,
}

impl PropertyPreferencesDraft {
    pub(crate) fn sync(&mut self, project: &ObjectPropertyPreferences) {
        if self.uncommitted {
            return;
        }
        if let Some((expected, remaining)) = self.awaiting.as_mut() {
            if expected != project && *remaining > 0 {
                *remaining -= 1;
                return;
            }
            self.awaiting = None;
        }
        if &self.preferences != project {
            self.preferences.clone_from(project);
        }
    }

    pub(crate) fn take_commit(&mut self) -> Option<Value> {
        if !self.uncommitted {
            return None;
        }
        self.uncommitted = false;
        self.awaiting = Some((self.preferences.clone(), COMMIT_WAIT_SYNCS));
        Some(json!({"object_property_preferences":self.preferences}))
    }

    fn is_pinned(&self, id: &str) -> bool {
        self.preferences.pinned.iter().any(|pinned| pinned == id)
    }

    fn toggle_pin(&mut self, id: &str) {
        if self.is_pinned(id) {
            self.preferences.pinned.retain(|pinned| pinned != id);
        } else {
            self.preferences.pinned.push(id.to_string());
        }
        self.uncommitted = true;
    }

    fn note_recent(&mut self, id: &str) {
        if self
            .preferences
            .recent
            .first()
            .is_some_and(|recent| recent == id)
        {
            return;
        }
        self.preferences.recent.retain(|recent| recent != id);
        self.preferences.recent.insert(0, id.to_string());
        self.preferences.recent.truncate(MAX_RECENT_PROPERTIES);
        self.uncommitted = true;
    }
}

fn find_item<'a>(items: &'a [PropertyItem], id: &str) -> Option<&'a PropertyItem> {
    items.iter().find(|item| item.id == id)
}

/// The previous or next entry, wrapping around. A pinned selection steps through the pins, which
/// is how to review a chosen set; anything else steps through its own group.
pub(crate) fn step_item<'a>(
    items: &'a [PropertyItem],
    preferences: &ObjectPropertyPreferences,
    selected: &str,
    forward: bool,
) -> Option<&'a str> {
    let pinned = preferences
        .pinned
        .iter()
        .filter_map(|id| find_item(items, id))
        .collect::<Vec<_>>();
    let sequence = if pinned.iter().any(|item| item.id == selected) {
        pinned
    } else {
        let group = find_item(items, selected).map(|item| &item.group);
        items
            .iter()
            .filter(|item| group.is_none_or(|group| &item.group == group))
            .collect()
    };
    if sequence.is_empty() {
        return None;
    }
    let target = match sequence.iter().position(|item| item.id == selected) {
        Some(position) if forward => (position + 1) % sequence.len(),
        Some(position) => (position + sequence.len() - 1) % sequence.len(),
        None if forward => 0,
        None => sequence.len() - 1,
    };
    let id = sequence[target].id.as_str();
    (id != selected).then_some(id)
}

/// Matches anywhere in the name and ignores case, so `h3` finds `pos_B7_H3`.
pub(crate) fn matches_search(item: &PropertyItem, search: &str) -> bool {
    let search = search.trim().to_lowercase();
    search.is_empty() || item.label.to_lowercase().contains(&search)
}

enum RowAction {
    Select,
    TogglePin,
}

fn item_row(
    ui: &mut egui::Ui,
    item: &PropertyItem,
    selected: bool,
    pinned: bool,
) -> Option<RowAction> {
    ui.horizontal(|ui| {
        let star = ui
            .small_button(if pinned { "★" } else { "☆" })
            .on_hover_text(if pinned { "Unpin" } else { "Pin to the top" });
        if star.clicked() {
            return Some(RowAction::TogglePin);
        }
        ui.selectable_label(selected, &item.label)
            .clicked()
            .then_some(RowAction::Select)
    })
    .inner
}

/// Draws one row: label, previous, picker, next. Returns the entry newly chosen, if any.
/// `[` and `]` also step, unless a text field has the keyboard.
pub(crate) fn show(
    ui: &mut egui::Ui,
    id_salt: &str,
    label: &str,
    items: &[PropertyItem],
    selected: &str,
    preferences: &mut PropertyPreferencesDraft,
) -> Option<String> {
    let mut chosen = None;
    let mut explicit = false;
    let id = ui.make_persistent_id(("property-picker", id_salt));
    ui.horizontal(|ui| {
        ui.label(label);
        let step_hint = "Step through your pins, or this group. Shortcut:";
        if ui
            .small_button("◀")
            .on_hover_text(format!("{step_hint} ["))
            .clicked()
        {
            chosen =
                step_item(items, &preferences.preferences, selected, false).map(str::to_string);
        }
        let mut pending = None;
        let open = egui::ComboBox::from_id_salt(id)
            .selected_text(if selected.is_empty() {
                "Choose a property"
            } else {
                selected
            })
            .height(420.0)
            .close_behavior(egui::PopupCloseBehavior::CloseOnClickOutside)
            .show_ui(ui, |ui| {
                ui.set_min_width(280.0);
                pending = popup_contents(ui, id, items, selected, preferences);
                if pending.is_some() {
                    ui.close();
                }
            })
            .inner
            .is_some();
        if !open {
            ui.data_mut(|data| {
                data.remove::<String>(id);
                data.remove::<bool>(id.with("focused"));
            });
        }
        if let Some(entry) = pending {
            chosen = Some(entry);
            explicit = true;
        }
        if ui
            .small_button("▶")
            .on_hover_text(format!("{step_hint} ]"))
            .clicked()
        {
            chosen = step_item(items, &preferences.preferences, selected, true).map(str::to_string);
        }
    });
    if !ui.ctx().wants_keyboard_input() {
        let (previous, next) = ui.input(|input| {
            (
                input.key_pressed(egui::Key::OpenBracket),
                input.key_pressed(egui::Key::CloseBracket),
            )
        });
        if previous || next {
            chosen = step_item(items, &preferences.preferences, selected, next).map(str::to_string);
        }
    }
    let chosen = chosen.filter(|entry| entry != selected)?;
    // Only deliberate picks count as recent, or stepping would flush the list immediately.
    if explicit {
        preferences.note_recent(&chosen);
    }
    Some(chosen)
}

/// Numeric columns as a marker and a statistic when they follow that pattern, so comparing markers
/// keeps the statistic; otherwise a single picker. Returns whether `property` changed.
pub(crate) fn show_numeric(
    ui: &mut egui::Ui,
    id_salt: &str,
    label: &str,
    columns: &[String],
    property: &mut String,
    preferences: &mut PropertyPreferencesDraft,
) -> bool {
    let markers = MarkerColumns::from_columns(columns);
    if !markers.is_useful() {
        let items = classify_properties(columns);
        return match show(ui, id_salt, label, &items, property, preferences) {
            Some(next) => {
                *property = next;
                true
            }
            None => false,
        };
    }
    let (entry, statistic) = markers.locate(property);
    let mut changed = false;
    if let Some(next) = show(ui, id_salt, label, &markers.items(), &entry, preferences)
        && let Some(column) = markers.column_for(&next, statistic.as_deref())
    {
        *property = column;
        changed = true;
    }
    if let Some(statistic) = statistic {
        let mut next_statistic = statistic.clone();
        ui.horizontal(|ui| {
            ui.label("Statistic");
            egui::ComboBox::from_id_salt(ui.make_persistent_id(("property-statistic", id_salt)))
                .selected_text(statistic_label(&statistic))
                .show_ui(ui, |ui| {
                    for option in markers.statistics(&entry) {
                        let text = statistic_label(&option);
                        ui.selectable_value(&mut next_statistic, option, text);
                    }
                });
        });
        if !changed
            && next_statistic != statistic
            && let Some(column) = markers.column_for(&entry, Some(&next_statistic))
        {
            *property = column;
            changed = true;
        }
    }
    changed
}

fn popup_contents(
    ui: &mut egui::Ui,
    id: egui::Id,
    items: &[PropertyItem],
    selected: &str,
    preferences: &mut PropertyPreferencesDraft,
) -> Option<String> {
    let mut search = ui
        .data_mut(|data| data.get_temp::<String>(id))
        .unwrap_or_default();
    let response = ui.add(
        egui::TextEdit::singleline(&mut search)
            .hint_text("Search…")
            .desired_width(f32::INFINITY),
    );
    if !ui.data_mut(|data| data.get_temp::<bool>(id.with("focused")).unwrap_or(false)) {
        response.request_focus();
        ui.data_mut(|data| data.insert_temp(id.with("focused"), true));
    }
    ui.data_mut(|data| data.insert_temp(id, search.clone()));

    let mut pending = None;
    let mut toggled = None;
    let mut row = |ui: &mut egui::Ui, item: &PropertyItem, pinned: bool| match item_row(
        ui,
        item,
        item.id == selected,
        pinned,
    ) {
        Some(RowAction::Select) => pending = Some(item.id.clone()),
        Some(RowAction::TogglePin) => toggled = Some(item.id.clone()),
        None => {}
    };
    egui::ScrollArea::vertical()
        .max_height(380.0)
        .auto_shrink([false, true])
        .show(ui, |ui| {
            if !search.trim().is_empty() {
                let mut any = false;
                for item in items.iter().filter(|item| matches_search(item, &search)) {
                    any = true;
                    row(ui, item, preferences.is_pinned(&item.id));
                }
                if !any {
                    ui.weak("No matching properties.");
                }
                return;
            }
            for (title, ids) in [
                ("Pinned", &preferences.preferences.pinned),
                ("Recent", &preferences.preferences.recent),
            ] {
                let present = ids
                    .iter()
                    .filter_map(|id| find_item(items, id))
                    .collect::<Vec<_>>();
                if present.is_empty() {
                    continue;
                }
                ui.weak(title);
                for item in present {
                    row(ui, item, preferences.preferences.pinned.contains(&item.id));
                }
                ui.separator();
            }
            let mut groups = BTreeMap::<&PropertyGroup, Vec<&PropertyItem>>::new();
            for item in items {
                groups.entry(&item.group).or_default().push(item);
            }
            for (group, members) in groups {
                egui::CollapsingHeader::new(format!("{} ({})", group.title(), members.len()))
                    .id_salt((id, group.title()))
                    .default_open(!matches!(group, PropertyGroup::Other))
                    .show(ui, |ui| {
                        for item in members {
                            row(ui, item, preferences.preferences.pinned.contains(&item.id));
                        }
                    });
            }
        });
    if let Some(id) = toggled {
        preferences.toggle_pin(&id);
    }
    pending
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| (*value).to_string()).collect()
    }

    fn group_of(items: &[PropertyItem], id: &str) -> PropertyGroup {
        find_item(items, id).unwrap().group.clone()
    }

    fn preferences(pinned: &[&str], recent: &[&str]) -> ObjectPropertyPreferences {
        ObjectPropertyPreferences {
            pinned: names(pinned),
            recent: names(recent),
        }
    }

    fn key_press(key: egui::Key) -> egui::RawInput {
        egui::RawInput {
            events: vec![egui::Event::Key {
                key,
                physical_key: None,
                pressed: true,
                repeat: false,
                modifiers: egui::Modifiers::NONE,
            }],
            ..Default::default()
        }
    }

    fn draw(context: &egui::Context, input: egui::RawInput, selected: &str) -> Option<String> {
        let items = classify_properties(&names(&["pos_A", "pos_B", "pos_C"]));
        let mut draft = PropertyPreferencesDraft::default();
        let mut chosen = None;
        let _ = context.run(input, |context| {
            egui::CentralPanel::default().show(context, |ui| {
                chosen = show(ui, "test", "Property", &items, selected, &mut draft);
            });
        });
        chosen
    }

    #[test]
    fn the_widget_draws_and_bracket_keys_step() {
        let context = egui::Context::default();
        assert_eq!(draw(&context, Default::default(), "pos_A"), None);

        assert_eq!(
            draw(&context, key_press(egui::Key::CloseBracket), "pos_A").as_deref(),
            Some("pos_B")
        );
        assert_eq!(
            draw(&context, key_press(egui::Key::OpenBracket), "pos_A").as_deref(),
            Some("pos_C")
        );
        assert_eq!(draw(&context, key_press(egui::Key::A), "pos_A"), None);
    }

    #[test]
    fn the_open_popup_draws_every_section() {
        let context = egui::Context::default();
        let items = classify_properties(&names(&["pos_A", "pos_B", "pos_C", "bbox_x0", "cluster"]));
        let mut draft = PropertyPreferencesDraft::default();
        draft.toggle_pin("pos_B");
        draft.note_recent("cluster");
        let id = egui::Id::new("test");
        let mut searches = [
            String::new(),
            "pos".to_string(),
            "nothing matches".to_string(),
        ]
        .into_iter();
        for _ in 0..3 {
            let search = searches.next().unwrap();
            let _ = context.run(Default::default(), |context| {
                egui::CentralPanel::default().show(context, |ui| {
                    ui.data_mut(|data| data.insert_temp(id, search.clone()));
                    assert_eq!(popup_contents(ui, id, &items, "pos_A", &mut draft), None);
                });
            });
        }
    }

    #[test]
    fn marker_columns_parse_from_the_end_because_names_contain_underscores() {
        let parse = |name: &str| parse_marker_column(name);
        let pair =
            |marker: &str, statistic: &str| Some((marker.to_string(), statistic.to_string()));

        assert_eq!(parse("marker_DAPI_mean_intensity"), pair("DAPI", "mean"));
        assert_eq!(
            parse("marker_B7_H3_C_EPR27429_173_p90_intensity"),
            pair("B7_H3_C_EPR27429_173", "p90")
        );
        assert_eq!(
            parse("marker_a_SMA_C_1A4_median_intensity_log1p"),
            pair("a_SMA_C_1A4", "median_log1p")
        );
        assert_eq!(parse("median_raw__cd3"), pair("cd3", "median_raw"));
    }

    #[test]
    fn columns_that_only_resemble_markers_are_not_parsed() {
        for name in [
            "pos_CD3",
            "marker_unknown",
            "cell_area",
            "__cd3",
            "median__",
            "marker__intensity",
        ] {
            assert_eq!(parse_marker_column(name), None, "{name}");
        }
    }

    #[test]
    fn real_columns_land_in_useful_groups() {
        let items = classify_properties(&names(&[
            "celltype_rules",
            "celltype_cluster",
            "cluster",
            "agreement",
            "pos_CD3",
            "pos_CD4",
            "pos_B7_H3",
            "marker_DAPI_mean_intensity",
            "bbox_x0",
            "bbox_y0_image",
            "x_centroid",
            "y_centroid_image",
            "cell_id",
            "geojson_path",
            "image_scale_x_nm",
            "geometry",
            "cell_area",
        ]));

        assert_eq!(group_of(&items, "celltype_rules"), PropertyGroup::General);
        assert_eq!(group_of(&items, "agreement"), PropertyGroup::General);
        assert_eq!(group_of(&items, "cell_area"), PropertyGroup::General);
        assert_eq!(
            group_of(&items, "pos_CD3"),
            PropertyGroup::Prefixed("pos".to_string())
        );
        assert_eq!(
            group_of(&items, "marker_DAPI_mean_intensity"),
            PropertyGroup::Markers
        );
        for rare in [
            "bbox_x0",
            "bbox_y0_image",
            "x_centroid",
            "y_centroid_image",
            "cell_id",
            "geojson_path",
            "image_scale_x_nm",
            "geometry",
        ] {
            assert_eq!(group_of(&items, rare), PropertyGroup::Other, "{rare}");
        }
    }

    #[test]
    fn nothing_is_dropped_by_grouping() {
        let columns = names(&["a", "pos_x", "bbox_x0", "marker_A_mean_intensity"]);
        let items = classify_properties(&columns);
        assert_eq!(
            items.iter().map(|item| &item.id).collect::<Vec<_>>(),
            columns.iter().collect::<Vec<_>>()
        );
    }

    #[test]
    fn search_ignores_case_and_matches_inside_names() {
        let item = &classify_properties(&names(&["pos_B7_H3"]))[0];
        assert!(matches_search(item, "h3"));
        assert!(matches_search(item, "  B7 "));
        assert!(matches_search(item, ""));
        assert!(!matches_search(item, "cd3"));
    }

    #[test]
    fn stepping_stays_in_the_group_and_wraps() {
        let items = classify_properties(&names(&["pos_A", "pos_B", "pos_C", "cluster"]));
        let none = preferences(&[], &[]);

        assert_eq!(step_item(&items, &none, "pos_A", true), Some("pos_B"));
        assert_eq!(step_item(&items, &none, "pos_C", true), Some("pos_A"));
        assert_eq!(step_item(&items, &none, "pos_A", false), Some("pos_C"));
        assert_eq!(
            step_item(&items, &none, "cluster", true),
            None,
            "alone in its group"
        );
    }

    #[test]
    fn a_pinned_selection_steps_through_the_pins_in_pin_order() {
        let items = classify_properties(&names(&["pos_A", "pos_B", "pos_C", "cluster"]));
        let pins = preferences(&["cluster", "pos_C", "gone"], &[]);

        assert_eq!(step_item(&items, &pins, "cluster", true), Some("pos_C"));
        assert_eq!(
            step_item(&items, &pins, "pos_C", true),
            Some("cluster"),
            "missing pins are skipped"
        );
        assert_eq!(
            step_item(&items, &pins, "pos_A", true),
            Some("pos_B"),
            "unpinned stays in its group"
        );
    }

    #[test]
    fn stepping_from_nothing_starts_at_an_end() {
        let items = classify_properties(&names(&["a", "b"]));
        let none = preferences(&[], &[]);
        assert_eq!(step_item(&items, &none, "", true), Some("a"));
        assert_eq!(step_item(&items, &none, "", false), Some("b"));
        assert_eq!(step_item(&[], &none, "", true), None);
    }

    #[test]
    fn switching_marker_keeps_the_statistic_when_it_exists() {
        let columns = MarkerColumns::from_columns(&names(&[
            "marker_CD3_mean_intensity",
            "marker_CD3_p90_intensity",
            "marker_CD4_mean_intensity",
            "cell_area",
        ]));

        assert!(columns.is_useful());
        assert_eq!(
            columns.locate("marker_CD3_p90_intensity"),
            ("CD3".to_string(), Some("p90".to_string()))
        );
        assert_eq!(
            columns.column_for("CD3", Some("p90")).as_deref(),
            Some("marker_CD3_p90_intensity")
        );
        assert_eq!(
            columns.column_for("CD4", Some("p90")).as_deref(),
            Some("marker_CD4_mean_intensity"),
            "CD4 has no p90, so it falls back rather than failing"
        );
        assert_eq!(
            columns.column_for("cell_area", Some("p90")).as_deref(),
            Some("cell_area")
        );
        assert_eq!(columns.locate("cell_area"), ("cell_area".to_string(), None));
        assert_eq!(columns.column_for("missing", None), None);
    }

    #[test]
    fn marker_items_offer_each_marker_once_plus_other_numeric_columns() {
        let columns = MarkerColumns::from_columns(&names(&[
            "marker_CD3_mean_intensity",
            "marker_CD3_p90_intensity",
            "marker_CD4_mean_intensity",
            "cell_area",
        ]));
        let ids = columns
            .items()
            .into_iter()
            .map(|item| item.id)
            .collect::<Vec<_>>();
        assert_eq!(ids, names(&["CD3", "CD4", "cell_area"]));
        assert_eq!(columns.statistics("CD3"), names(&["mean", "p90"]));
    }

    #[test]
    fn one_marker_does_not_justify_a_second_control() {
        assert!(
            !MarkerColumns::from_columns(&names(&["marker_CD3_mean_intensity", "a"])).is_useful()
        );
    }

    #[test]
    fn log_statistics_read_clearly() {
        assert_eq!(statistic_label("median_log1p"), "median (log1p)");
        assert_eq!(statistic_label("p90"), "p90");
    }

    #[test]
    fn pins_toggle_and_are_committed_once() {
        let mut draft = PropertyPreferencesDraft::default();
        draft.toggle_pin("pos_A");
        assert!(draft.is_pinned("pos_A"));

        let commit = draft.take_commit().expect("an edit commits");
        assert_eq!(
            commit["object_property_preferences"]["pinned"],
            json!(["pos_A"])
        );
        assert!(draft.take_commit().is_none());

        draft.toggle_pin("pos_A");
        assert!(!draft.is_pinned("pos_A"));
    }

    #[test]
    fn recents_are_ordered_deduplicated_and_bounded() {
        let mut draft = PropertyPreferencesDraft::default();
        for index in 0..10 {
            draft.note_recent(&format!("p{index}"));
        }
        draft.note_recent("p7");

        let recent = &draft.preferences.recent;
        assert_eq!(recent.len(), MAX_RECENT_PROPERTIES);
        assert_eq!(recent[0], "p7");
        assert_eq!(recent[1], "p9");
        assert_eq!(recent.iter().filter(|id| *id == "p7").count(), 1);
    }

    #[test]
    fn a_stale_project_snapshot_does_not_undo_a_new_pin() {
        let mut draft = PropertyPreferencesDraft::default();
        draft.toggle_pin("pos_A");
        let before = preferences(&[], &[]);

        draft.sync(&before);
        assert!(draft.is_pinned("pos_A"), "uncommitted edits win");

        draft.take_commit();
        draft.sync(&before);
        assert!(draft.is_pinned("pos_A"), "the commit is still travelling");

        draft.sync(&preferences(&["pos_A"], &[]));
        draft.sync(&preferences(&["pos_A", "pos_B"], &[]));
        assert!(
            draft.is_pinned("pos_B"),
            "once applied, the project is the source of truth"
        );
    }
}
