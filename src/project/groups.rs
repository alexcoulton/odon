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
