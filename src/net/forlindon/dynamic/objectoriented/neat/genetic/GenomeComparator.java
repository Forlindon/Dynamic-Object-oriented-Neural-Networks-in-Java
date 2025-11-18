package net.forlindon.dynamic.objectoriented.neat.genetic;

import java.util.Comparator;

public class GenomeComparator implements Comparator<Genome> {

    @Override
    public int compare(Genome o1, Genome o2) {
        return Double.compare(o2.calcDelta(o1), o1.calcDelta(o2));
    }
}
