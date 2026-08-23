//! Renderer-side snapshots and projection adapters for the control actor.
//!
//! Production code in this module may observe native UI state or apply an actor projection, but it
//! must not implement application command semantics. A small set of direct mutation helpers is
//! retained under `cfg(test)` while the older GUI characterization tests move to actor fixtures.

mod channels;
mod layers;
mod masks;
mod objects;
mod resources;
mod view;
mod viewports;
