package net.forlindon.dynamic.objectoriented.neat.genetic;

public interface Genotype {

    int inov();

    void mutate();

    double getMutationRate();

}
