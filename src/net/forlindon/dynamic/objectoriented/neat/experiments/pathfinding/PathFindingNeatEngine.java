package net.forlindon.dynamic.objectoriented.neat.experiments.pathfinding;

import net.forlindon.dynamic.objectoriented.neat.NeatEngine;
import net.forlindon.dynamic.objectoriented.neat.genetic.Genome;
import net.forlindon.dynamic.objectoriented.neat.genetic.InnovationSource;
import net.forlindon.dynamic.objectoriented.neat.genetic.Species;
import net.forlindon.dynamic.objectoriented.neat.genetic.SpeciesManager;
import net.forlindon.dynamic.objectoriented.neat.knot.ReluNeatKnot;
import net.forlindon.dynamic.objectoriented.neat.layer.NeatLinearLayer;
import net.forlindon.dynamic.objectoriented.neat.layer.SequentialNeatLayer;
import net.forlindon.dynamic.objectoriented.neat.visuals.NetWrapper;
import net.forlindon.dynamic.objectoriented.neural.networks.tensor.activation.TanhTensor;

public class PathFindingNeatEngine extends NeatEngine {

    Game game;
    NetWrapper netWrapper;

    public PathFindingNeatEngine(Game game, NetWrapper netWrapper) {
        SpeciesManager.setDefaultActivation(TanhTensor::new);
        this.innovationSource = new InnovationSource();
        SequentialNeatLayer sequentialNeatLayer = new SequentialNeatLayer(innovationSource);
        sequentialNeatLayer.addLayer(innovationSource1 -> new NeatLinearLayer(6,innovationSource1, ReluNeatKnot::new));
        sequentialNeatLayer.addLayer(innovationSource1 -> new NeatLinearLayer(6,innovationSource1, ReluNeatKnot::new));
        sequentialNeatLayer.addLayer(innovationSource1 -> new NeatLinearLayer(2,innovationSource1, ReluNeatKnot::new));
        sequentialNeatLayer.fullyConnect();
        Genome g = new Genome(sequentialNeatLayer);
        this.speciesManager = new SpeciesManager(2000);

        Species species1 = new Species(this.speciesManager, g);
        species1.add(g);
        this.speciesManager.init(species1);
        this.game = game;
        this.speciesManager.SPECIATION_BORDER = 2;
        this.speciesManager.c1 = 1.1;
        this.speciesManager.c3 = 0.6;
        this.netWrapper = netWrapper;
        this.report = 1;
    }

    @Override
    public Genome run() {
        this.speciesManager.getSpecies().forEach(species -> species.genomes.forEach(genome -> genome.setFitness(0)));
        game.evaluate(this.speciesManager.getSpecies(), this.phenoTypeBuilder);
        do {
        }
        while (game.t.isRunning());
        Genome fittest = this.getSpeciesManager().getFittest().copy();
        this.getSpeciesManager().sort();
        this.getSpeciesManager().removeWeakest();
        this.getSpeciesManager().populate();
        this.getSpeciesManager().mutate();
        if (++gen % report == 0) {
            System.out.printf("Gen: %d, Av: %s, Max: %g, Max Delta: %.5g, Species: %d, Inov: %s, Manager: %s, Deaths: %d\n", gen, speciesManager.averageFitness(), fittest.getFitness(), speciesManager.getMaxDelta(), speciesManager.getNumOfSpecies(), innovationSource, speciesManager, game.players.stream().filter(player -> player.dead).count());
            netWrapper.set(this.phenoTypeBuilder.build(this.speciesManager.getFittest()).get());
        }
        return fittest;
    }
}
