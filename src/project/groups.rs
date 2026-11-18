use crate::data::project_config::{
    ProjectAnnotationGroup, ProjectAnnotationGroupMember, ProjectChannelGroup,
    ProjectChannelGroupMember, ProjectLayerGroups,
};

pub fn next_group_id(existing: &[u64]) -> u64 {
    existing
        .iter()
        .copied()
        .max()
        .unwrap_or(0)
        .wrapping_add(1)
        .max(1)
}

pub fn find_channel_group<'a>(
    groups: &'a ProjectLayerGroups,
    group_id: u64,
) -> Option<&'a ProjectChannelGroup> {
    groups.channel_groups.iter().find(|g| g.id == group_id)
}

pub fn find_channel_member<'a>(
    groups: &'a ProjectLayerGroups,
    channel_name: &str,
) -> Option<&'a ProjectChannelGroupMember> {
    groups.channel_members.get(channel_name)
}

pub fn effective_channel_color_rgb(
    groups: &ProjectLayerGroups,
    channel_name: &str,
    channel_color_rgb: [u8; 3],
) -> [u8; 3] {
    let Some(member) = find_channel_member(groups, channel_name) else {
        return channel_color_rgb;
    };
    if !member.inherit_color {
        return channel_color_rgb;
    }
    let Some(group) = find_channel_group(groups, member.group_id) else {
        return channel_color_rgb;
    };
    group.color_rgb
}

/// Whether a group's channels use its colour. Counts every member, including any hidden by the
/// channel search or absent from the open image, because the setting applies to the whole group.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ChannelGroupColorInheritance {
    Empty,
    None,
    Mixed,
    All,
}

pub fn channel_group_color_inheritance(
    groups: &ProjectLayerGroups,
    group_id: u64,
) -> ChannelGroupColorInheritance {
    let mut inheriting = false;
    let mut independent = false;
    for member in groups
        .channel_members
        .values()
        .filter(|member| member.group_id == group_id)
    {
        inheriting |= member.inherit_color;
        independent |= !member.inherit_color;
    }
    match (inheriting, independent) {
        (false, false) => ChannelGroupColorInheritance::Empty,
        (false, true) => ChannelGroupColorInheritance::None,
        (true, true) => ChannelGroupColorInheritance::Mixed,
        (true, false) => ChannelGroupColorInheritance::All,
    }
}

/// Channels keep their own colours while inheriting, so turning this off restores them.
pub fn set_channel_group_color_inheritance(
    groups: &mut ProjectLayerGroups,
    group_id: u64,
    inherit_color: bool,
) -> bool {
    let mut changed = false;
    for member in groups
        .channel_members
        .values_mut()
        .filter(|member| member.group_id == group_id)
    {
        changed |= member.inherit_color != inherit_color;
        member.inherit_color = inherit_color;
    }
    changed
}

pub fn find_annotation_group<'a>(
    groups: &'a ProjectLayerGroups,
    group_id: u64,
) -> Option<&'a ProjectAnnotationGroup> {
    groups.annotation_groups.iter().find(|g| g.id == group_id)
}

pub fn find_annotation_member<'a>(
    groups: &'a ProjectLayerGroups,
    layer_id: u64,
) -> Option<&'a ProjectAnnotationGroupMember> {
    groups.annotation_members.get(&layer_id)
}

pub fn effective_annotation_tint(
    groups: &ProjectLayerGroups,
    layer_id: u64,
) -> Option<([u8; 3], f32)> {
    let Some(member) = find_annotation_member(groups, layer_id) else {
        return None;
    };
    if !member.inherit_tint {
        return None;
    }
    let Some(group) = find_annotation_group(groups, member.group_id) else {
        return None;
    };
    let Some(rgb) = group.tint_rgb else {
        return None;
    };
    Some((rgb, group.tint_strength.clamp(0.0, 1.0)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn member(group_id: u64, inherit_color: bool) -> ProjectChannelGroupMember {
        ProjectChannelGroupMember {
            group_id,
            inherit_color,
        }
    }

    fn groups(members: &[(&str, u64, bool)]) -> ProjectLayerGroups {
        ProjectLayerGroups {
            channel_groups: vec![ProjectChannelGroup {
                id: 1,
                name: "T cells".to_string(),
                expanded: true,
                color_rgb: [10, 20, 30],
            }],
            channel_members: members
                .iter()
                .map(|(name, group_id, inherit)| (name.to_string(), member(*group_id, *inherit)))
                .collect::<HashMap<_, _>>(),
            ..ProjectLayerGroups::default()
        }
    }

    #[test]
    fn group_inheritance_summarises_every_member() {
        use ChannelGroupColorInheritance as State;
        let summary =
            |members: &[(&str, u64, bool)]| channel_group_color_inheritance(&groups(members), 1);

        assert_eq!(summary(&[]), State::Empty);
        assert_eq!(summary(&[("other", 2, true)]), State::Empty);
        assert_eq!(summary(&[("a", 1, false), ("b", 1, false)]), State::None);
        assert_eq!(summary(&[("a", 1, true), ("b", 1, false)]), State::Mixed);
        assert_eq!(summary(&[("a", 1, true), ("b", 1, true)]), State::All);
    }

    #[test]
    fn setting_inheritance_changes_only_that_groups_members() {
        let mut config = groups(&[("a", 1, false), ("b", 1, true), ("other", 2, false)]);

        assert!(set_channel_group_color_inheritance(&mut config, 1, true));

        assert_eq!(
            channel_group_color_inheritance(&config, 1),
            ChannelGroupColorInheritance::All
        );
        assert!(!config.channel_members["other"].inherit_color);
    }

    #[test]
    fn setting_the_current_state_reports_no_change() {
        let mut config = groups(&[("a", 1, true)]);
        assert!(!set_channel_group_color_inheritance(&mut config, 1, true));
        assert!(!set_channel_group_color_inheritance(&mut config, 9, false));
    }

    #[test]
    fn turning_inheritance_off_restores_each_channels_own_colour() {
        let own_color = [200, 100, 50];
        let mut config = groups(&[("a", 1, false)]);
        assert_eq!(
            effective_channel_color_rgb(&config, "a", own_color),
            own_color
        );

        set_channel_group_color_inheritance(&mut config, 1, true);
        assert_eq!(
            effective_channel_color_rgb(&config, "a", own_color),
            [10, 20, 30]
        );

        set_channel_group_color_inheritance(&mut config, 1, false);
        assert_eq!(
            effective_channel_color_rgb(&config, "a", own_color),
            own_color
        );
    }
}
