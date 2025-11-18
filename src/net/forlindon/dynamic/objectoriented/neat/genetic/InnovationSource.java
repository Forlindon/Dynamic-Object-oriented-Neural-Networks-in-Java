package net.forlindon.dynamic.objectoriented.neat.genetic;

import java.util.Set;
import java.util.TreeSet;

public class InnovationSource {

    private final Set<Integer> RANGE;

    public InnovationSource() {
        this.RANGE = new TreeSet<>();
    }

    public void add(int val) {
        this.RANGE.add(val);
    }

    @Override
    public String toString() {
        return String.valueOf(this.RANGE.size());
    }

    public int getNext() {
        int r = Integer.MIN_VALUE + this.RANGE.size() - 1;
        this.add(r);
        return r;
    }

    public boolean isInRange(int i) {
        return this.RANGE.contains(i);
    }
}
