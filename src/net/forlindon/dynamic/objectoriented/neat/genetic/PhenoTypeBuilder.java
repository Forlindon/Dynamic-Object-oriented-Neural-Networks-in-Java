package net.forlindon.dynamic.objectoriented.neat.genetic;

import net.forlindon.dynamic.objectoriented.neat.connection.BaseNeatConnection;
import net.forlindon.dynamic.objectoriented.neat.knot.BaseNeatKnot;
import net.forlindon.dynamic.objectoriented.neat.layer.BaseNeatLayer;
import net.forlindon.dynamic.objectoriented.neat.layer.SequentialNeatLayer;

import java.util.Map;

public class PhenoTypeBuilder {

    public PhenoType build(Genome g) {
        Map<Integer, BaseNeatKnot> nodes = g.NODES;

        SequentialNeatLayer net = new SequentialNeatLayer(g.innovationSource);

        for (Integer i : g.LAYER) {
            net.add(new BaseNeatLayer(g.innovationSource, i));
        }

        for (BaseNeatKnot baseNeatKnot : nodes.values()) {
            net.add(baseNeatKnot);
        }
        return new PhenoType(net);
    }

}
