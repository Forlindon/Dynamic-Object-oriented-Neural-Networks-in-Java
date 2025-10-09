package net.forlindon.dynamic.objectoriented.neat.knot;

import net.forlindon.dynamic.objectoriented.neural.networks.knot.Knot;

import java.util.Comparator;

public class KnotComparator implements Comparator<Knot> {

    @Override
    public int compare(Knot o1, Knot o2) {
        return Integer.compare(o1.id(), o2.id());
    }
}
