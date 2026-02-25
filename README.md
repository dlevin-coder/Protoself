1 Introduction

Artificial systems capable of flexible, context-sensitive behavior typically rely on internal states that
modulate perception and action beyond immediate sensory input. In many architectures, such internal
states are treated implicitly—as parameters, hidden activations, or control variables—without
being explicitly modeled as objects that the system can itself observe or manipulate. This raises a
structural question: under what conditions can an artificial system maintain an internal representation
of itself that participates in processing on the same footing as other objects?

A common response is to introduce a central agent, controller, or persistent subject-like component
responsible for coordinating internal processes. An alternative approach, explored in this
work, is to treat self-representation not as an agent or control center, but as an internal object dynamically
reconstructed from the system’s own activity. On this view, the system does not require
a privileged subject-level entity; instead, self-related structure emerges from recursive relations
among internal representations.

We distinguish between two functional roles: observer mechanisms, which integrate sensory
input and internally generated signals, and a self-object, defined as an internally represented state
corresponding to the current configuration of the observer mechanisms themselves. The self-object
is continuously reconstructed and may enter the same processing pathways as externally derived
objects.

Reflexive dynamics arise when the self-object itself becomes the target of observation, creating
a recursive structure in which internal states are both generators and objects of processing. This
recursion is implemented at the architectural level and does not presuppose introspection, selfawareness,
or phenomenological subjectivity.

We present a spiking neural implementation of this framework and evaluate it
using a set of behavioral tests designed to probe three minimal properties expected of a self-object:
(i) modulation of identical sensory inputs by internal state, (ii) persistence of internal state in
the absence of external stimulation, and (iii) controllable global dynamics analogous to affective
valence. The goal is not to model consciousness or subjective experience, but to demonstrate that a
self-object, construed as a dynamically maintained internal representation, is sufficient to support
these properties.


2 Model / Methods

2.1 Architecture Overview

The architecture consists of:
• Observer mechanisms: integrate sensory and internally generated signals.
• Self-object module: dynamically reconstructed internal representation corresponding to
current observer configuration.
Reflexive dynamics emerge when the self-object enters the same processing stream as external
inputs, creating a recursive structure.

2.2 Spiking Neural Network Implementation
• Built using Brian2.
• Neurons are grouped into functional populations corresponding to observer and self-object
modules.
• Short-term memory and global activation dynamics are implemented via recurrent connections.
