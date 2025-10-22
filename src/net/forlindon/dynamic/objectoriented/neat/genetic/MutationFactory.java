package net.forlindon.dynamic.objectoriented.neat.genetic;

import net.forlindon.dynamic.objectoriented.neat.connection.BaseNeatConnection;
import net.forlindon.dynamic.objectoriented.neat.knot.BaseNeatKnot;

public class MutationFactory {

    Genome genome;

    public MutationFactory(Genome genome) {
        this.genome = genome;
    }

    public Genome mutate() {
        Genome genome = this.genome.copy();
        genome.copyValues();
        for (BaseNeatConnection baseNeatConnection : genome.GENES.values()) {
            if (Math.random() < baseNeatConnection.getMutationRate()) {
                baseNeatConnection.mutate();
            }
        }
        for (BaseNeatKnot baseNeatKnot : genome.NODES.values()) {
            if (Math.random() < baseNeatKnot.getMutationRate()) {
                baseNeatKnot.mutate();
            }
        }
        if (Math.random() < genome.getMutationRate()) {
            int idx = genome.GENES.keySet().stream().toList().get((int)(Math.random() * genome.GENES.size()));
            BaseNeatConnection baseNeatConnection = genome.GENES.get(idx);

            int id = baseNeatConnection.src().id();
            int idIdx = genome.LAYER.indexOf(id) + 1;

            if (idIdx > 0 && idIdx < genome.LAYER.size()-1) {

                id = genome.innovationSource.getNext();
                genome.LAYER.add(idIdx, id);

                BaseNeatKnot mid = new BaseNeatKnot(genome.innovationSource, id);
                BaseNeatConnection midToDest = new BaseNeatConnection(genome.innovationSource, mid, baseNeatConnection.dest());

                baseNeatConnection.setDest(mid);
                mid.add(midToDest);
                genome.GENES.put(midToDest.getInnovationNumber(), midToDest);

                genome.NODES.put(id, mid);
            }
        }
        return genome;
    }

}
