//! Temporary compatibility boundary for frame-driven handlers not yet removed by the actor
//! migration. New semantic commands belong in the central actor, not in this module.

mod analysis;
mod channels;
mod layers;
mod masks;
mod objects;
mod resources;
mod view;
mod viewports;
