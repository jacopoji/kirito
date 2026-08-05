use std::fmt::Debug;

use crate::{Character, PotionMode, player::PlayerContext};

/// A service to handle character-related incoming requests.
pub trait CharacterService: Debug {
    /// Gets the currently in use [`Character`].
    fn character(&self) -> Option<&Character>;

    /// Sets a new `character` to be used.
    fn update_character(&mut self, character: Option<Character>);

    /// Updates `player_context` with information from the currently in use `[Character]`.
    fn apply_character(&self, player_context: &mut PlayerContext);
}

#[derive(Debug, Default)]
pub struct DefaultCharacterService {
    character: Option<Character>,
}

impl CharacterService for DefaultCharacterService {
    fn character(&self) -> Option<&Character> {
        self.character.as_ref()
    }

    fn update_character(&mut self, character: Option<Character>) {
        self.character = character;
    }

    fn apply_character(&self, player_context: &mut PlayerContext) {
        player_context.reset();
        if let Some(character) = self.character.as_ref() {
            player_context.config.link_key_timing_millis = character.link_key_timing_millis;
            player_context.config.has_extended_teleport_range =
                character.has_extended_teleport_range;
            player_context.config.disable_double_jumping = character.disable_double_jumping;
            player_context.config.disable_adjusting = character.disable_adjusting;
            player_context.config.disable_teleport_on_fall = character.disable_teleport_on_fall;
            player_context.config.disable_grapple_on_double_jumping =
                character.disable_grapple_on_double_jumping;
            player_context.config.up_jump_is_flight = character.up_jump_is_flight;
            player_context.config.up_jump_specific_key_should_jump =
                character.up_jump_specific_key_should_jump;
            player_context.config.interact_key = character.interact_key.key.into();
            player_context.config.grappling_key = character.ropelift_key.map(|key| key.key.into());
            player_context.config.teleport_key = character.teleport_key.map(|key| key.key.into());
            player_context.config.blink_key = character.blink_key.map(|key| key.key.into());
            player_context.config.jump_key = character.jump_key.key.into();
            player_context.config.up_jump_key = character.up_jump_key.map(|key| key.key.into());
            player_context.config.cash_shop_key = character.cash_shop_key.map(|key| key.key.into());
            player_context.config.familiar_key =
                character.familiar_menu_key.map(|key| key.key.into());
            player_context.config.familiar_buff_key = character
                .familiar_buff_key
                .enabled
                .then_some(character.familiar_buff_key.key.into());
        assert_eq!(state.config.inventory_key, Some(KeyKind::I));
        assert_eq!(state.config.to_town_key, Some(KeyKind::M));
        assert_eq!(state.config.change_channel_key, Some(KeyKind::L));
        assert_eq!(state.config.potion_key, KeyKind::P);
        assert_eq!(state.config.use_potion_below_percent, Some(0.5));
        assert_eq!(state.config.update_health_millis, Some(3000));
    }

    #[test]
    fn update_from_character_maps_familiar_buff_key_when_enabled() {
        let mut service = DefaultCharacterService::default();
        let character = Character {
            familiar_buff_key: KeyBindingConfiguration {
                key: KeyBinding::F,
                enabled: true,
            },
            ..mock_character()
        };
        service.update_character(Some(character));

        let mut state = PlayerContext::default();
        service.apply_character(&mut state);

        assert_eq!(state.config.familiar_buff_key, Some(KeyKind::F));
    }
}
