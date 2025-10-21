package net.forlindon.dynamic.objectoriented.neat.genetic;

import net.forlindon.dynamic.objectoriented.neat.connection.BaseNeatConnection;
import net.forlindon.dynamic.objectoriented.neat.knot.BaseNeatKnot;
import net.forlindon.dynamic.objectoriented.neat.layer.SequentialNeatLayer;

import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

public class PhenoTypeBuilder {

    public PhenoType build(Genome g) {
        Map<Integer, BaseNeatKnot> nodes = g.NODES.values().stream().collect(Collectors.toMap(
                BaseNeatKnot::getInnovationNumber,
                BaseNeatKnot::copy
        ));
        List<BaseNeatConnection> genes = g.GENES.values().stream().map(BaseNeatConnection::copy).toList();
        genes.forEach(x -> x.setSrc(nodes.get(x.src().getInnovationNumber())));
        genes.forEach(x -> x.setDest(nodes.get(x.dest().getInnovationNumber())));
        SequentialNeatLayer net = new SequentialNeatLayer(g.innovationSource);
        for (BaseNeatKnot baseNeatKnot : nodes.values()) {
            net.add(baseNeatKnot);
        }
        for (BaseNeatConnection baseNeatConnection : genes) {
            BaseNeatKnot baseNeatKnot = nodes.get(baseNeatConnection.src().getInnovationNumber());
            baseNeatKnot.add(baseNeatConnection);
        }
        return new PhenoType(net);
    }

}
