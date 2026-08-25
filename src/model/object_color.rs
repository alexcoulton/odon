//! Renderer-independent object colour-mapping contracts and colour interpolation.

use serde::{Deserialize, Serialize};

const DOMAIN_EPSILON: f64 = 1.0e-12;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum ObjectColorMapping {
    Single,
    Categorical {
        property: String,
    },
    Continuous {
        property: String,
        #[serde(default)]
        palette: ContinuousPalette,
        #[serde(default)]
        domain: ContinuousDomain,
        #[serde(default)]
        scale: ContinuousScale,
        #[serde(default)]
        reverse: bool,
        #[serde(default)]
        out_of_range: OutOfRangeMode,
        #[serde(default)]
        missing_color_rgb: Option<[u8; 3]>,
    },
}

impl Default for ObjectColorMapping {
    fn default() -> Self {
        Self::Single
    }
}

impl ObjectColorMapping {
    pub fn categorical(property: impl Into<String>) -> Self {
        Self::Categorical {
            property: property.into(),
        }
    }

    pub fn property(&self) -> Option<&str> {
        match self {
            Self::Single => None,
            Self::Categorical { property } | Self::Continuous { property, .. } => Some(property),
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        let Some(property) = self.property() else {
            return Ok(());
        };
        if property.trim().is_empty() {
            return Err("object color mapping property must not be empty".to_string());
        }
        let Self::Continuous {
            palette,
            domain,
            scale,
            ..
        } = self
        else {
            return Ok(());
        };
        palette.validate()?;
        domain.validate()?;
        if *scale == ContinuousScale::Log10
            && let ContinuousDomain::Fixed([minimum, maximum]) = domain
            && (*minimum <= 0.0 || *maximum <= 0.0)
        {
            return Err("log10 object color domains must be greater than zero".to_string());
        }
        Ok(())
    }

    pub fn continuous_config(&self) -> Option<ContinuousColorConfig<'_>> {
        match self {
            Self::Continuous {
                palette,
                domain,
                scale,
                reverse,
                out_of_range,
                missing_color_rgb,
                ..
            } => Some(ContinuousColorConfig {
                palette,
                domain,
                scale: *scale,
                reverse: *reverse,
                out_of_range: *out_of_range,
                missing_color_rgb: *missing_color_rgb,
            }),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum ContinuousDomain {
    Automatic(String),
    Fixed([f64; 2]),
}

impl Default for ContinuousDomain {
    fn default() -> Self {
        Self::Automatic("auto".to_string())
    }
}

impl ContinuousDomain {
    pub fn validate(&self) -> Result<(), String> {
        match self {
            Self::Automatic(value) if value == "auto" => Ok(()),
            Self::Automatic(_) => {
                Err("object color domain must be 'auto' or [min, max]".to_string())
            }
            Self::Fixed([minimum, maximum])
                if minimum.is_finite() && maximum.is_finite() && minimum < maximum =>
            {
                Ok(())
            }
            Self::Fixed(_) => Err(
                "object color domain must contain two finite numbers with min < max".to_string(),
            ),
        }
    }

    pub fn fixed(&self) -> Option<[f64; 2]> {
        match self {
            Self::Fixed(domain) => Some(*domain),
            Self::Automatic(_) => None,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ContinuousScale {
    #[default]
    Linear,
    Log10,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum OutOfRangeMode {
    #[default]
    Clamp,
    Hide,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum ContinuousPalette {
    Named(String),
    Custom(Vec<ContinuousColorStop>),
}

impl Default for ContinuousPalette {
    fn default() -> Self {
        Self::Named("viridis".to_string())
    }
}

impl ContinuousPalette {
    pub const NAMED: [&'static str; 7] = [
        "viridis", "magma", "plasma", "inferno", "cividis", "turbo", "gray",
    ];

    pub fn validate(&self) -> Result<(), String> {
        match self {
            Self::Named(name) if Self::NAMED.contains(&name.as_str()) => Ok(()),
            Self::Named(_) => Err(format!(
                "unknown object color palette; expected one of {}",
                Self::NAMED.join(", ")
            )),
            Self::Custom(stops) => validate_custom_stops(stops),
        }
    }

    pub fn color_rgb(&self, position: f64) -> [u8; 3] {
        match self {
            Self::Named(name) => interpolate_stops(named_palette_stops(name), position),
            Self::Custom(stops) => interpolate_custom_stops(stops, position),
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct ContinuousColorStop {
    pub position: f64,
    pub color_rgb: [u8; 3],
}

#[derive(Debug, Clone, Copy)]
pub struct ContinuousColorConfig<'a> {
    pub palette: &'a ContinuousPalette,
    pub domain: &'a ContinuousDomain,
    pub scale: ContinuousScale,
    pub reverse: bool,
    pub out_of_range: OutOfRangeMode,
    pub missing_color_rgb: Option<[u8; 3]>,
}

impl ContinuousColorConfig<'_> {
    pub fn color_rgba(&self, value: Option<f64>, resolved_domain: [f64; 2]) -> [u8; 4] {
        let Some(value) = value.filter(|value| value.is_finite()) else {
            return self.missing_rgba();
        };
        let [minimum, maximum] = resolved_domain;
        if !minimum.is_finite() || !maximum.is_finite() {
            return self.missing_rgba();
        }
        let transformed = match self.scale {
            ContinuousScale::Linear => Some((value, minimum, maximum)),
            ContinuousScale::Log10 if value > 0.0 && minimum > 0.0 && maximum > 0.0 => {
                Some((value.log10(), minimum.log10(), maximum.log10()))
            }
            ContinuousScale::Log10 => None,
        };
        let Some((value, minimum, maximum)) = transformed else {
            return self.missing_rgba();
        };
        if (maximum - minimum).abs() <= DOMAIN_EPSILON {
            return self.palette_rgba(0.5);
        }
        let mut position = (value - minimum) / (maximum - minimum);
        if !(0.0..=1.0).contains(&position) {
            if self.out_of_range == OutOfRangeMode::Hide {
                return [0, 0, 0, 0];
            }
            position = position.clamp(0.0, 1.0);
        }
        self.palette_rgba(position)
    }

    fn palette_rgba(&self, position: f64) -> [u8; 4] {
        let position = if self.reverse {
            1.0 - position
        } else {
            position
        };
        let [red, green, blue] = self.palette.color_rgb(position);
        [red, green, blue, 255]
    }

    fn missing_rgba(&self) -> [u8; 4] {
        self.missing_color_rgb
            .map(|[red, green, blue]| [red, green, blue, 255])
            .unwrap_or([0, 0, 0, 0])
    }
}

fn validate_custom_stops(stops: &[ContinuousColorStop]) -> Result<(), String> {
    if stops.len() < 2 {
        return Err("custom object color palettes require at least two stops".to_string());
    }
    if stops.len() > 256 {
        return Err("custom object color palettes support at most 256 stops".to_string());
    }
    if stops
        .iter()
        .any(|stop| !stop.position.is_finite() || !(0.0..=1.0).contains(&stop.position))
    {
        return Err("custom object color stop positions must be between 0 and 1".to_string());
    }
    if stops
        .windows(2)
        .any(|pair| pair[0].position >= pair[1].position)
    {
        return Err("custom object color stops must be strictly ordered".to_string());
    }
    if stops.first().is_none_or(|stop| stop.position != 0.0)
        || stops.last().is_none_or(|stop| stop.position != 1.0)
    {
        return Err("custom object color palettes must include positions 0 and 1".to_string());
    }
    Ok(())
}

fn interpolate_custom_stops(stops: &[ContinuousColorStop], position: f64) -> [u8; 3] {
    if stops.is_empty() {
        return [255, 255, 255];
    }
    let position = position.clamp(0.0, 1.0);
    let upper = stops.partition_point(|stop| stop.position < position);
    if upper == 0 {
        return stops[0].color_rgb;
    }
    if upper >= stops.len() {
        return stops[stops.len() - 1].color_rgb;
    }
    let low = stops[upper - 1];
    let high = stops[upper];
    let denominator = high.position - low.position;
    let fraction = if denominator.abs() <= DOMAIN_EPSILON {
        0.0
    } else {
        (position - low.position) / denominator
    };
    interpolate_rgb(low.color_rgb, high.color_rgb, fraction)
}

fn interpolate_stops(stops: &[[u8; 3]], position: f64) -> [u8; 3] {
    if stops.is_empty() {
        return [255, 255, 255];
    }
    if stops.len() == 1 {
        return stops[0];
    }
    let scaled = position.clamp(0.0, 1.0) * (stops.len() - 1) as f64;
    let low = scaled.floor() as usize;
    let high = scaled.ceil() as usize;
    interpolate_rgb(stops[low], stops[high], scaled - low as f64)
}

fn interpolate_rgb(low: [u8; 3], high: [u8; 3], fraction: f64) -> [u8; 3] {
    let fraction = fraction.clamp(0.0, 1.0);
    std::array::from_fn(|index| {
        ((low[index] as f64 * (1.0 - fraction) + high[index] as f64 * fraction).round()) as u8
    })
}

fn named_palette_stops(name: &str) -> &'static [[u8; 3]] {
    match name {
        "magma" => &[
            [0, 0, 4],
            [28, 16, 68],
            [79, 18, 123],
            [129, 37, 129],
            [181, 54, 122],
            [229, 80, 100],
            [251, 135, 97],
            [254, 194, 135],
            [252, 253, 191],
        ],
        "plasma" => &[
            [13, 8, 135],
            [75, 3, 161],
            [125, 3, 168],
            [168, 34, 150],
            [203, 70, 121],
            [229, 107, 93],
            [248, 148, 65],
            [253, 195, 40],
            [240, 249, 33],
        ],
        "inferno" => &[
            [0, 0, 4],
            [31, 12, 72],
            [85, 15, 109],
            [136, 34, 106],
            [186, 54, 85],
            [227, 89, 51],
            [249, 140, 10],
            [249, 201, 50],
            [252, 255, 164],
        ],
        "cividis" => &[
            [0, 32, 77],
            [31, 54, 110],
            [60, 75, 117],
            [87, 95, 120],
            [113, 116, 120],
            [142, 137, 120],
            [174, 160, 116],
            [207, 185, 107],
            [255, 233, 69],
        ],
        "turbo" => &[
            [48, 18, 59],
            [67, 62, 133],
            [54, 113, 173],
            [36, 164, 165],
            [74, 201, 108],
            [164, 221, 57],
            [239, 190, 31],
            [245, 105, 23],
            [122, 4, 3],
        ],
        "gray" => &[[0, 0, 0], [255, 255, 255]],
        _ => &[
            [68, 1, 84],
            [71, 44, 122],
            [59, 82, 139],
            [44, 113, 142],
            [33, 144, 141],
            [39, 173, 129],
            [92, 200, 99],
            [170, 220, 50],
            [253, 231, 37],
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mapping(
        palette: ContinuousPalette,
        domain: ContinuousDomain,
        reverse: bool,
        out_of_range: OutOfRangeMode,
    ) -> ObjectColorMapping {
        ObjectColorMapping::Continuous {
            property: "score".to_string(),
            palette,
            domain,
            scale: ContinuousScale::Linear,
            reverse,
            out_of_range,
            missing_color_rgb: None,
        }
    }

    #[test]
    fn named_palette_maps_domain_endpoints_and_reverse() {
        let normal = mapping(
            ContinuousPalette::Named("viridis".to_string()),
            ContinuousDomain::Fixed([0.0, 10.0]),
            false,
            OutOfRangeMode::Clamp,
        );
        let reverse = mapping(
            ContinuousPalette::Named("viridis".to_string()),
            ContinuousDomain::Fixed([0.0, 10.0]),
            true,
            OutOfRangeMode::Clamp,
        );
        let normal = normal.continuous_config().unwrap();
        let reverse = reverse.continuous_config().unwrap();
        assert_eq!(normal.color_rgba(Some(0.0), [0.0, 10.0]), [68, 1, 84, 255]);
        assert_eq!(
            normal.color_rgba(Some(10.0), [0.0, 10.0]),
            [253, 231, 37, 255]
        );
        assert_eq!(
            reverse.color_rgba(Some(0.0), [0.0, 10.0]),
            [253, 231, 37, 255]
        );
    }

    #[test]
    fn custom_palette_interpolates_and_handles_constant_domain() {
        let mapping = mapping(
            ContinuousPalette::Custom(vec![
                ContinuousColorStop {
                    position: 0.0,
                    color_rgb: [0, 10, 20],
                },
                ContinuousColorStop {
                    position: 1.0,
                    color_rgb: [100, 110, 120],
                },
            ]),
            ContinuousDomain::Automatic("auto".to_string()),
            false,
            OutOfRangeMode::Clamp,
        );
        mapping.validate().unwrap();
        let config = mapping.continuous_config().unwrap();
        assert_eq!(config.color_rgba(Some(5.0), [0.0, 10.0]), [50, 60, 70, 255]);
        assert_eq!(config.color_rgba(Some(5.0), [5.0, 5.0]), [50, 60, 70, 255]);
    }

    #[test]
    fn hide_and_missing_semantics_are_explicit() {
        let mapping = mapping(
            ContinuousPalette::Named("gray".to_string()),
            ContinuousDomain::Fixed([0.0, 1.0]),
            false,
            OutOfRangeMode::Hide,
        );
        let config = mapping.continuous_config().unwrap();
        assert_eq!(config.color_rgba(Some(-1.0), [0.0, 1.0]), [0, 0, 0, 0]);
        assert_eq!(config.color_rgba(None, [0.0, 1.0]), [0, 0, 0, 0]);
        assert_eq!(config.color_rgba(Some(f64::NAN), [0.0, 1.0]), [0, 0, 0, 0]);
    }

    #[test]
    fn log_scale_maps_decades_and_rejects_nonpositive_values() {
        let mapping = ObjectColorMapping::Continuous {
            property: "score".to_string(),
            palette: ContinuousPalette::Named("gray".to_string()),
            domain: ContinuousDomain::Fixed([1.0, 100.0]),
            scale: ContinuousScale::Log10,
            reverse: false,
            out_of_range: OutOfRangeMode::Clamp,
            missing_color_rgb: None,
        };
        mapping.validate().unwrap();
        let config = mapping.continuous_config().unwrap();
        assert_eq!(
            config.color_rgba(Some(10.0), [1.0, 100.0]),
            [128, 128, 128, 255]
        );
        assert_eq!(config.color_rgba(Some(0.0), [1.0, 100.0]), [0, 0, 0, 0]);
    }

    #[test]
    fn invalid_mapping_is_rejected() {
        assert!(
            mapping(
                ContinuousPalette::Named("rainbow".to_string()),
                ContinuousDomain::Fixed([0.0, 1.0]),
                false,
                OutOfRangeMode::Clamp,
            )
            .validate()
            .is_err()
        );
        assert!(
            mapping(
                ContinuousPalette::Named("viridis".to_string()),
                ContinuousDomain::Fixed([1.0, 1.0]),
                false,
                OutOfRangeMode::Clamp,
            )
            .validate()
            .is_err()
        );
    }

    #[test]
    fn mapping_round_trips_through_contract_json() {
        let value = serde_json::json!({
            "mode":"continuous",
            "property":"mean_channel_1",
            "palette":"viridis",
            "domain":[4000.0,42000.0],
            "scale":"linear",
            "reverse":false,
            "out_of_range":"clamp",
            "missing_color_rgb":null,
        });
        let mapping: ObjectColorMapping = serde_json::from_value(value.clone()).unwrap();
        mapping.validate().unwrap();
        assert_eq!(serde_json::to_value(mapping).unwrap(), value);
    }
}
