package net.forlindon.dynamic.objectoriented.neat.genetic;

import java.util.Comparator;

public class GenomeComparator implements Comparator<Genome> {

    SpeciesManager speciesManager;

    public GenomeComparator(SpeciesManager speciesManager) {
        this.speciesManager = speciesManager;
    }

    @Override
    public int compare(Genome o1, Genome o2) {
        return Double.compare(o2.calcDelta(o1, speciesManager.c1, speciesManager.c2, speciesManager.c3), o1.calcDelta(o2, speciesManager.c1, speciesManager.c2, speciesManager.c3));
    }
}
