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
        return this.RANGE.toString();
    }

    public int getNext() {
        int r = this.RANGE.size();
        this.add(r);
        return r;
    }

    public boolean isInRange(int i) {
        return this.RANGE.contains(i);
    }
}
