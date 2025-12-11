package net.forlindon.dynamic.objectoriented.neat.experiments.tictactoe;

import net.forlindon.dynamic.objectoriented.neat.NeatEngin;
import net.forlindon.dynamic.objectoriented.neat.connection.BaseNeatConnection;
import net.forlindon.dynamic.objectoriented.neat.genetic.Genome;
import net.forlindon.dynamic.objectoriented.neat.genetic.Species;
import net.forlindon.dynamic.objectoriented.neat.genetic.SpeciesManager;
import net.forlindon.dynamic.objectoriented.neat.knot.BaseNeatKnot;
import net.forlindon.dynamic.objectoriented.neat.knot.NoiseNeatKnot;
import net.forlindon.dynamic.objectoriented.neat.knot.ReluNeatKnot;
import net.forlindon.dynamic.objectoriented.neat.knot.SigNeatKnot;
import net.forlindon.dynamic.objectoriented.neat.layer.NeatLinearLayer;
import net.forlindon.dynamic.objectoriented.neat.layer.SequentialNeatLayer;
import net.forlindon.dynamic.objectoriented.neural.networks.tensor.activation.ReluTensor;

import java.util.List;

public class Engin extends NeatEngin {

    Environment environment;

    public Engin() {
        super();
        SpeciesManager.setDefaultActivation(ReluTensor::new);
        environment = new Environment();
        this.report = 10;
        SequentialNeatLayer net = new SequentialNeatLayer(innovationSource);
        net.addLayer(src -> new NeatLinearLayer(9, src, BaseNeatKnot::new));
        net.addLayer(src -> new NeatLinearLayer(9, src, SigNeatKnot::new));
        net.fullyConnect(BaseNeatConnection::new);
        this.speciesManager = new SpeciesManager(200);
        this.speciesManager.c1 = 0.4;
        this.speciesManager.c2 = 0.4;
        this.speciesManager.c3 = 1;
        this.speciesManager.SPECIATION_BORDER = 1.2;

        Genome g = new Genome(net);
        Species species = new Species(speciesManager, g);
        species.add(g);

        speciesManager.init(species);

        this.speciesManager.mutate();
    }

    @Override
    public void evaluate() {
        List<Agent> agents = this.speciesManager.genomes().stream().map(genome -> new Agent(genome, this.phenoTypeBuilder)).toList();
        if (gen < 100) {
            this.environment.evaluateAll(agents);
        }
        else {
            this.environment.evaluateAdvanced(agents);
        }
    }
}
