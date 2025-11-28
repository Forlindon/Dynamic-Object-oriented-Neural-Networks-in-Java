package net.forlindon.dynamic.objectoriented.neat.knot;

import java.util.Comparator;

public class BaseNeatKnotComparator implements Comparator<BaseNeatKnot> {

    @Override
    public int compare(BaseNeatKnot o1, BaseNeatKnot o2) {
        return Integer.compare(o1.inov(),o2.inov());
    }
}
