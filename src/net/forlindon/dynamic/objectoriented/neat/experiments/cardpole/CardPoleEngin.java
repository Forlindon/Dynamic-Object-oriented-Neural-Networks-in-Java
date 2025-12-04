package net.forlindon.dynamic.objectoriented.neat.experiments.cardpole;

import net.forlindon.dynamic.objectoriented.neat.NeatEngin;
import net.forlindon.dynamic.objectoriented.neat.connection.BaseNeatConnection;
import net.forlindon.dynamic.objectoriented.neat.experiments.Pair;
import net.forlindon.dynamic.objectoriented.neat.genetic.*;
import net.forlindon.dynamic.objectoriented.neat.knot.BaseNeatKnot;
import net.forlindon.dynamic.objectoriented.neat.knot.ReluNeatKnot;
import net.forlindon.dynamic.objectoriented.neat.knot.TanhNeatKnot;
import net.forlindon.dynamic.objectoriented.neat.layer.NeatLinearLayer;
import net.forlindon.dynamic.objectoriented.neat.layer.SequentialNeatLayer;

import java.util.List;

public class CardPoleEngin extends NeatEngin {

    CardPoleEnvironment environment;

    public CardPoleEngin() {
        super();
        environment = new CardPoleEnvironment();
        this.report = 1;
        SequentialNeatLayer net = new SequentialNeatLayer(innovationSource);
        net.addLayer(src -> new NeatLinearLayer(4, src, BaseNeatKnot::new));
        net.addLayer(src -> new NeatLinearLayer(128, src, ReluNeatKnot::new));
        net.addLayer(src -> new NeatLinearLayer(1, src, TanhNeatKnot::new));
        net.fullyConnect(BaseNeatConnection::new);
        this.speciesManager = new SpeciesManager(100);

        Genome g = new Genome(net);
        Species species = new Species(speciesManager, g);
        species.add(g);

        speciesManager.init(species);

        this.speciesManager.mutate();
    }

    @Override
    public void evaluate() {
        List<Agent> pairs = this.speciesManager.getSpecies().stream().map(species -> species.genomes.stream().map(genome -> new Pair<>(genome, this.phenoTypeBuilder.build(genome))).toList()).flatMap(List::stream).map(Agent::new).toList();

        do {
            for (Agent agent : pairs) {
                this.environment.runAgent(agent);
            }
        } while (pairs.stream().filter(agent -> agent.done).count() != pairs.size());

        pairs.forEach(agent -> {
            agent.agent.a().setFitness(agent.episodes);
            agent.reset();
        });
    }
}
