package net.forlindon.dynamic.objectoriented.neat.experiments.artifical.life.sim.visuals;

import net.forlindon.dynamic.objectoriented.neat.experiments.artifical.life.sim.visuals.objects.Object;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;

public class Frame extends JFrame {

    Game game;

    public Frame() {
        setTitle("Test-World");
        setLayout(new GridLayout());
        setDefaultCloseOperation(EXIT_ON_CLOSE);
        setVisible(true);
        this.game = new Game();
        setContentPane(game);
        pack();
        setResizable(false);
        // Timer t = new Timer(20, this::tick);
        // t.start();
        gameLoop();
    }

    public void gameLoop() {
        Thread game = new Thread(() -> {
            final int fps = 50;
            final long frameTime = 1000 / fps;

            while (true) {
                long start = System.currentTimeMillis();

                this.game.world.tickEverything();

                SwingUtilities.invokeLater(this.game::repaint);

                long sleep = frameTime - (System.currentTimeMillis() - start);
                if (sleep > 0) {
                    try {
                        Thread.sleep(sleep);
                    }
                    catch (InterruptedException e) {}
                }
            }
        });
        game.setDaemon(true);
        game.start();
    }

    private void tick(ActionEvent e) {
        this.game.world.tickEverything();
        this.game.world.OBJECTS.sort(Object::compareTo);
        repaint();
    }
}
