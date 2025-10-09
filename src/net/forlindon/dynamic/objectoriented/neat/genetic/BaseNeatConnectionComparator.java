package net.forlindon.dynamic.objectoriented.neat.genetic;

import net.forlindon.dynamic.objectoriented.neat.connection.BaseNeatConnection;

import java.util.Comparator;

public class BaseNeatConnectionComparator implements Comparator<BaseNeatConnection> {
    @Override
    public int compare(BaseNeatConnection o1, BaseNeatConnection o2) {
        return Integer.compare(o1.getInnovationNumber(),o2.getInnovationNumber());
    }
}
