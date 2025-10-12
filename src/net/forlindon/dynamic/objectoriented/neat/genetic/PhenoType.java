package net.forlindon.dynamic.objectoriented.neat.genetic;

import net.forlindon.dynamic.objectoriented.neat.layer.BaseNeatLayer;

public class PhenoType implements Comparable<PhenoType> {

    double fitness = 0;
    BaseNeatLayer net;


    public PhenoType(BaseNeatLayer net) {
        this.net = net;
    }

    public void setFitness(double fitness) {
        this.fitness = fitness;
    }

    @Override
    public int compareTo(PhenoType o) {
        return Double.compare(this.fitness, o.fitness);
    }
}
