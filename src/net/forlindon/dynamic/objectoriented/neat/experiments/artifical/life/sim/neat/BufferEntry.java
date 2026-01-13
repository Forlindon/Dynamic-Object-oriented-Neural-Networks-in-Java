package net.forlindon.dynamic.objectoriented.neat.experiments.artifical.life.sim.neat;

import net.forlindon.dynamic.objectoriented.neat.genetic.PhenoType;

public record BufferEntry(PhenoType phenoType, double[] input, double[] out) {}
