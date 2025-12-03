package net.forlindon.dynamic.objectoriented.neat.experiments.pathfinding;

import net.forlindon.dynamic.objectoriented.neat.NeatEngin;
import net.forlindon.dynamic.objectoriented.neat.experiments.Pair;
import net.forlindon.dynamic.objectoriented.neat.genetic.Genome;
import net.forlindon.dynamic.objectoriented.neat.genetic.PhenoType;
import net.forlindon.dynamic.objectoriented.neat.genetic.PhenoTypeBuilder;
import net.forlindon.dynamic.objectoriented.neat.genetic.Species;
import net.forlindon.dynamic.objectoriented.neat.visuals.NetWrapper;
import net.forlindon.dynamic.objectoriented.neat.visuals.NetworkDisplay;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.util.ArrayList;

public class Visuals extends JFrame {

    Game game;
    NeatEngin engin;
    NetWrapper netWrapper = new NetWrapper(null);

    public Visuals() {
        this.setSize(1400,700);
        game = new Game();
        this.setLayout(new GridLayout());
        this.add(game);
        this.add(new NetworkDisplay(netWrapper));
        this.engin = new PathFindingNeatEngin(game, netWrapper);
        this.setDefaultCloseOperation(EXIT_ON_CLOSE);
        this.setResizable(false);
        this.setVisible(true);
        train();
    }

    public void train() {
        new Thread(() -> {

            for (int i = 0; i < 500; i++) {
                this.setTitle(String.format("Gen: %d", engin.getGen()));
                Genome fitness = engin.run();
                netWrapper.set(engin.getPhenoTypeBuilder().build(fitness).get());
            }

        }).start();
    }
}

class Game extends JPanel {

    private static final int offsetX = 10;
    private static final int offsetY = 10;

    public static final int PLAYER_SIZE = 40;

    private final int POPULATION_SIZE = 300;
    java.util.List<Player> players = new ArrayList<>(POPULATION_SIZE);
    private final Rectangle wall = new Rectangle(this.x+this.wallOffsetX, this.y+this.wallOffsetY, this.wallWidth,this.wallHeight);

    private final int wallWidth = 30;
    private final int wallHeight = 300;
    private final int wallOffsetX = 350;
    private final int wallOffsetY = 210;

    private int width;
    private int height;

    private final int x = offsetX;
    private final int y = offsetY;

    Timer t = new Timer(0, this::event);

    int roundCounter;

    public boolean ready = true;

    public Game() {
    }

    public void evaluate(java.util.List<Species> species, PhenoTypeBuilder phenoTypeBuilder) {
        this.players.clear();
        ready = false;
        roundCounter = 0;
        this.width = getWidth()-2*offsetX;
        this.height = getHeight()-2*offsetY;
        for (Species species1 : species) {
            Color color = new Color((int) (255*Math.random()), (int) (255*Math.random()), (int) (255*Math.random()));
            for (Genome g : species1.genomes) {
                Player p = new Player(PLAYER_SIZE, PLAYER_SIZE, color);
                p.add(x+offsetX,y +(int)(Math.random()*(height-p.h())));
                p.setAgent(new Pair<>(g, phenoTypeBuilder.build(g)));
                players.add(p);
            }
        }
        t.start();
    }

    @Override
    protected void paintComponent(Graphics g) {
        super.paintComponent(g);

        Graphics2D g2d = (Graphics2D) g;

        g2d.setColor(Color.DARK_GRAY);
        g2d.fillRect(x,y,width,height);

        g2d.setColor(Color.BLACK);

        g2d.fill(this.wall);
        this.players.forEach(player -> player.draw(g2d));
    }

    public void event(ActionEvent e) {
        if (roundCounter++ > 256) {
            roundCounter = 0;
            for (Player p : this.players) {
                Genome g = p.agent.a();
                double delta = width*0.75-(p.x()+p.w());
                g.setFitness(g.getFitness()-delta-(g.getNODES().size()-7));
            }
            ready = true;
            this.t.stop();
            return;
        }
        this.players.forEach(player -> player.updateValues(this.wall));
        this.players.forEach(this::movePlayer);
        repaint();
    }

    public void movePlayer(Player player) {

        if (player.dead) return;

        Genome agent = player.agent.a();
        double dF = 0;
        
        if (this.wall.intersects(player.x() + player.velocityX(), player.y(), player.w(), player.h())) {
            dF -= 2000;
            player.dead = true;
        }
        if (this.wall.intersects(player.x(), player.y() + player.velocityY(), player.w(), player.h())) {
            dF -= 1500;
            player.dead = true;
        }
        if (isInBound(player, player.velocityX(), player.velocityY())) {
            player.add(player.velocityX(), player.velocityY());
        }
        else {
            if (!isInBoundX(player, player.velocityX())) {
                player.setVelocityX(0);
                dF -= 300;
            }
            else {
                player.setVelocityY(0);
                dF -= 600;
            }
            player.dead = true;
        }
        agent.setFitness(agent.getFitness()+dF);
    }
    public boolean isInBound(Player player, double dx, double dy) {
        return isInBoundX(player, dx) && isInBoundY(player, dy);
    }
    public boolean isInBoundX(Player player, double dx) {
        return player.x() - offsetX + player.w() + dx < width && player.x() + dx > offsetX;
    }
    public boolean isInBoundY(Player player, double dy) {
        return player.y() - offsetY + player.h() + dy < height && player.y() + dy > offsetY;
    }

    public boolean ready() {
        return this.ready;
    }
}

class Player {

    Rectangle body;
    double velocityX, velocityY;
    Pair<Genome, PhenoType> agent;
    double[] inp = new double[6];
    double[] out = new double[2];
    Color color;
    boolean dead = false;

    public Player(int w, int h, Color color) {
        this.body = new Rectangle(w,h);
        this.color = color;
    }

    public void addX(double dx) {
        this.body.x += (int)Math.round(dx);
    }

    public void addY(double dy) {
        this.body.y += (int)Math.round(dy);
    }

    public void add(double dx, double dy) {
        addX(dx);
        addY(dy);
    }

    public void add(int d) {
        addX(d);
        addY(d);
    }

    public int x() {
        return (int) this.body.getX();
    }

    public int y() {
        return (int) this.body.getY();
    }

    public int w() {
        return (int) this.body.getWidth();
    }

    public int h() {
        return (int) this.body.getHeight();
    }

    public boolean intersects(Rectangle p) {
        return this.body.intersects(p);
    }

    public void draw(Graphics2D g) {
        if (this.dead) return;
        g.setColor(this.color);
        g.fill(this.body);
        g.setColor(Color.BLACK);
    }

    public void setVelocityX(double v) {
        this.velocityX = v;
    }

    public void setVelocityY(double v) {
        this.velocityY = v;
    }

    public double velocityX() {
        return velocityX;
    }

    public double velocityY() {
        return velocityY;
    }

    public void setAgent(Pair<Genome, PhenoType> agent) {
        this.agent = agent;
    }

    public void updateValues(Rectangle wall) {
        if (this.agent == null || dead) return;
        this.inp[0] = x();
        this.inp[1] = y();
        this.inp[2] = velocityX();
        this.inp[3] = velocityY();
        this.inp[4] = wall.x;
        this.inp[5] = wall.y;

        this.agent.b().forward(inp,out);
        this.setVelocityX(out[0]*3);
        this.setVelocityY(out[1]*3);
    }

    public Color getColor() {
        return color;
    }
}

