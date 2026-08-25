//! Cross-platform realization of actor-owned command shortcut descriptors.

use eframe::egui;
use serde_json::Value;

use super::CommandPresentationInvocation;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ShortcutPlatform {
    MacOs,
    Other,
}

impl ShortcutPlatform {
    const fn current() -> Self {
        if cfg!(target_os = "macos") {
            Self::MacOs
        } else {
            Self::Other
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ParsedShortcut {
    key: egui::Key,
    modifiers: egui::Modifiers,
}

/// Consume and resolve the first available command shortcut in the active actor projection.
///
/// The actor rejects overlapping shortcuts before they reach this layer. The native realization
/// still consumes the exact key event so legacy widgets cannot dispatch the same action again.
#[cfg_attr(target_os = "macos", allow(dead_code))]
pub(crate) fn resolve(
    ctx: &egui::Context,
    command_surface: &Value,
) -> Option<CommandPresentationInvocation> {
    if ctx.wants_keyboard_input() {
        return None;
    }
    command_surface
        .get("commands")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter(|command| command_available(command))
        .find_map(|command| {
            let shortcut = command.get("shortcut")?;
            if !consume(ctx, Some(shortcut)) {
                return None;
            }
            let command_id = command.get("id").and_then(Value::as_str)?;
            Some(command_invocation(command_id, command))
        })
}

/// Consume one exact descriptor shortcut. Used by the command palette as well as command actions.
pub(crate) fn consume(ctx: &egui::Context, shortcut: Option<&Value>) -> bool {
    let Some(parsed) = shortcut.and_then(|value| parse(value, ShortcutPlatform::current()).ok())
    else {
        return false;
    };
    ctx.input_mut(|input| {
        let mut consumed = false;
        input.events.retain(|event| {
            let matches = matches!(
                event,
                egui::Event::Key {
                    key,
                    pressed: true,
                    modifiers,
                    ..
                } if *key == parsed.key && modifiers.matches_exact(parsed.modifiers)
            );
            if matches {
                consumed = true;
            }
            !matches
        });
        consumed
    })
}

pub(crate) fn neutral_label(shortcut: Option<&Value>) -> Option<String> {
    let shortcut = shortcut?.as_object()?;
    let mut parts = shortcut
        .get("modifiers")?
        .as_array()?
        .iter()
        .filter_map(Value::as_str)
        .map(|modifier| match modifier {
            "primary" => "Primary".to_string(),
            "shift" => "Shift".to_string(),
            "alt" => "Alt".to_string(),
            "control" => "Ctrl".to_string(),
            "super" => "Super".to_string(),
            other => other.to_string(),
        })
        .collect::<Vec<_>>();
    parts.push(shortcut.get("key")?.as_str()?.to_ascii_uppercase());
    Some(parts.join("+"))
}

#[cfg_attr(target_os = "macos", allow(dead_code))]
fn command_available(command: &Value) -> bool {
    command
        .pointer("/state/visible")
        .and_then(Value::as_bool)
        .unwrap_or(true)
        && command
            .pointer("/state/enabled")
            .and_then(Value::as_bool)
            .unwrap_or(true)
}

#[cfg_attr(target_os = "macos", allow(dead_code))]
fn command_invocation(command_id: &str, command: &Value) -> CommandPresentationInvocation {
    let checked = command
        .pointer("/state/checked")
        .and_then(Value::as_bool)
        .map(|checked| !checked);
    CommandPresentationInvocation {
        command_id: command_id.to_string(),
        checked,
    }
}

fn parse(shortcut: &Value, platform: ShortcutPlatform) -> Result<ParsedShortcut, &'static str> {
    let key = shortcut
        .get("key")
        .and_then(Value::as_str)
        .and_then(egui_key)
        .ok_or("unsupported shortcut key")?;
    let mut modifiers = egui::Modifiers::NONE;
    for modifier in shortcut
        .get("modifiers")
        .and_then(Value::as_array)
        .ok_or("shortcut modifiers are missing")?
        .iter()
        .map(|value| value.as_str().ok_or("invalid shortcut modifier"))
    {
        modifiers |= match modifier? {
            "primary" => egui::Modifiers::COMMAND,
            "shift" => egui::Modifiers::SHIFT,
            "alt" => egui::Modifiers::ALT,
            "control" => egui::Modifiers::CTRL,
            "super" if platform == ShortcutPlatform::MacOs => egui::Modifiers::MAC_CMD,
            "super" => {
                return Err("the Super modifier is unavailable through egui on this platform");
            }
            _ => return Err("unsupported shortcut modifier"),
        };
    }
    Ok(ParsedShortcut { key, modifiers })
}

fn egui_key(key: &str) -> Option<egui::Key> {
    Some(match key.to_ascii_lowercase().as_str() {
        "a" => egui::Key::A,
        "b" => egui::Key::B,
        "c" => egui::Key::C,
        "d" => egui::Key::D,
        "e" => egui::Key::E,
        "f" => egui::Key::F,
        "g" => egui::Key::G,
        "h" => egui::Key::H,
        "i" => egui::Key::I,
        "j" => egui::Key::J,
        "k" => egui::Key::K,
        "l" => egui::Key::L,
        "m" => egui::Key::M,
        "n" => egui::Key::N,
        "o" => egui::Key::O,
        "p" => egui::Key::P,
        "q" => egui::Key::Q,
        "r" => egui::Key::R,
        "s" => egui::Key::S,
        "t" => egui::Key::T,
        "u" => egui::Key::U,
        "v" => egui::Key::V,
        "w" => egui::Key::W,
        "x" => egui::Key::X,
        "y" => egui::Key::Y,
        "z" => egui::Key::Z,
        "comma" => egui::Key::Comma,
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn input_for(key: egui::Key, modifiers: egui::Modifiers) -> egui::RawInput {
        egui::RawInput {
            events: vec![egui::Event::Key {
                key,
                physical_key: None,
                pressed: true,
                repeat: false,
                modifiers,
            }],
            modifiers,
            ..Default::default()
        }
    }

    fn primary_modifiers() -> egui::Modifiers {
        if cfg!(target_os = "macos") {
            egui::Modifiers {
                mac_cmd: true,
                command: true,
                ..Default::default()
            }
        } else {
            egui::Modifiers {
                ctrl: true,
                command: true,
                ..Default::default()
            }
        }
    }

    #[test]
    fn descriptor_shortcuts_resolve_available_commands_and_consume_the_event() {
        let context = egui::Context::default();
        let modifiers = primary_modifiers();
        context.begin_pass(input_for(egui::Key::K, modifiers));
        let surface = json!({"commands":[{
            "id":"extension:org.example/check",
            "shortcut":{"key":"k","modifiers":["primary"]},
            "state":{"visible":true,"enabled":true,"checked":false}
        }]});
        assert_eq!(
            resolve(&context, &surface),
            Some(CommandPresentationInvocation {
                command_id: "extension:org.example/check".to_string(),
                checked: Some(true),
            })
        );
        assert!(!context.input(|input| input.key_pressed(egui::Key::K)));
        let _ = context.end_pass();
    }

    #[test]
    fn focused_text_input_suppresses_descriptor_shortcuts() {
        let context = egui::Context::default();
        let mut text = String::new();
        context.begin_pass(Default::default());
        egui::CentralPanel::default().show(&context, |ui| {
            ui.text_edit_singleline(&mut text).request_focus();
        });
        let _ = context.end_pass();

        let modifiers = primary_modifiers();
        context.begin_pass(input_for(egui::Key::K, modifiers));
        assert!(context.wants_keyboard_input());
        let surface = json!({"commands":[{
            "id":"extension:org.example/run",
            "shortcut":{"key":"k","modifiers":["primary"]},
            "state":{"visible":true,"enabled":true,"checked":null}
        }]});
        assert_eq!(resolve(&context, &surface), None);
        assert!(context.input(|input| input.key_pressed(egui::Key::K)));
        let _ = context.end_pass();
    }

    #[test]
    fn changed_shortcut_projection_takes_effect_without_stale_registration() {
        let context = egui::Context::default();
        let modifiers = primary_modifiers();
        let original = json!({"commands":[{
            "id":"extension:org.example/run",
            "shortcut":{"key":"k","modifiers":["primary"]},
            "state":{"visible":true,"enabled":true,"checked":null}
        }]});
        context.begin_pass(input_for(egui::Key::K, modifiers));
        assert!(resolve(&context, &original).is_some());
        let _ = context.end_pass();

        let changed = json!({"commands":[{
            "id":"extension:org.example/run",
            "shortcut":{"key":"j","modifiers":["primary"]},
            "state":{"visible":true,"enabled":true,"checked":null}
        }]});
        context.begin_pass(input_for(egui::Key::K, modifiers));
        assert_eq!(resolve(&context, &changed), None);
        assert!(context.input(|input| input.key_pressed(egui::Key::K)));
        let _ = context.end_pass();

        context.begin_pass(input_for(egui::Key::J, modifiers));
        assert_eq!(
            resolve(&context, &changed),
            Some(CommandPresentationInvocation {
                command_id: "extension:org.example/run".to_string(),
                checked: None,
            })
        );
        let _ = context.end_pass();
    }

    #[test]
    fn unavailable_commands_and_extra_modifiers_do_not_dispatch() {
        let context = egui::Context::default();
        let modifiers = if cfg!(target_os = "macos") {
            egui::Modifiers {
                mac_cmd: true,
                command: true,
                shift: true,
                ..Default::default()
            }
        } else {
            egui::Modifiers {
                ctrl: true,
                command: true,
                shift: true,
                ..Default::default()
            }
        };
        let surface = json!({"commands":[{
            "id":"app.settings.open",
            "shortcut":{"key":"comma","modifiers":["primary"]},
            "state":{"visible":true,"enabled":false,"checked":null}
        }]});
        context.begin_pass(input_for(egui::Key::Comma, modifiers));
        assert_eq!(resolve(&context, &surface), None);
        assert!(context.input(|input| input.key_pressed(egui::Key::Comma)));
        let _ = context.end_pass();
    }

    #[test]
    fn shortcut_parsing_reports_platform_specific_super_support() {
        let shortcut = json!({"key":"q","modifiers":["super"]});
        assert!(parse(&shortcut, ShortcutPlatform::MacOs).is_ok());
        assert_eq!(
            parse(&shortcut, ShortcutPlatform::Other),
            Err("the Super modifier is unavailable through egui on this platform")
        );
        assert_eq!(
            neutral_label(Some(&json!({"key":"p","modifiers":["primary","shift"]}))),
            Some("Primary+Shift+P".to_string())
        );
    }
}
