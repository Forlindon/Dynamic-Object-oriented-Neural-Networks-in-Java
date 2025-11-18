package net.forlindon.dynamic.objectoriented.neat.genetic;

import java.util.Comparator;

public class GenomeFitnessComparator implements Comparator<Genome> {
    @Override
    public int compare(Genome o1, Genome o2) {
        return Double.compare(o1.fitness,o2.fitness);
    }
}
