package net.forlindon.dynamic.objectoriented.neat.layer;

import net.forlindon.dynamic.objectoriented.neat.genetic.InnovationSource;
import net.forlindon.dynamic.objectoriented.neat.knot.BaseNeatKnot;

import java.util.function.BiFunction;

public class NeatLinearLayer extends BaseNeatLayer {

    public NeatLinearLayer(int n, InnovationSource paramSrc, BiFunction<InnovationSource, Integer, BaseNeatKnot> factory) {
        super(paramSrc, paramSrc.getNext());
        for (int i = 0; i < n; i++) {
            this.add(factory);
        }
    }

}
