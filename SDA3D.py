import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Constante gravitationnelle
G = 6.67430e-11

# Classe pour représenter chaque corps dans la simulation
class Body:
    def __init__(self, position, mass, velocity):
        self.position = np.array(position, dtype=float)
        self.mass = mass
        self.velocity = np.array(velocity, dtype=float)
        self.force = np.array([0.0, 0.0, 0.0], dtype=float)  # Force initialisée en tant que float

    def update_velocity(self, timestep):
        acceleration = self.force / self.mass
        self.velocity += acceleration * timestep

    def update_position(self, timestep):
        self.position += self.velocity * timestep

# Classe pour les nœuds de l'arbre octal
class OctreeNode:
    def __init__(self, center, size):
        self.center = center
        self.size = size
        self.children = [None] * 8
        self.is_leaf = True
        self.body = None
        self.total_mass = 0
        self.center_of_mass = [0, 0, 0]

    def get_child_index(self, body_pos):
        index = 0
        if body_pos[0] > self.center[0]: index |= 1
        if body_pos[1] > self.center[1]: index |= 2
        if body_pos[2] > self.center[2]: index |= 4
        return index




    def insert(self, body, depth=0):
        # Limite de profondeur pour éviter la récursion infinie
        if depth > 20:  
            return

        if self.is_leaf:
            if self.body is None:
                self.body = body
                self.total_mass = body.mass
                self.center_of_mass = body.position
            else:
                # Sauvegarder l'ancien corps et subdiviser
                existing_body = self.body
                self.body = None
                self.subdivide()

                # Réinsérer l'ancien corps et le nouveau corps dans les enfants
                self.insert_to_child(existing_body, depth + 1)
                self.insert_to_child(body, depth + 1)
        else:
            self.insert_to_child(body, depth + 1)

    def insert_to_child(self, body, depth):
        index = self.get_child_index(body.position)
        if self.children[index] is None:
            # Calcul du nouveau centre pour l'enfant
            new_center = [self.center[i] + self.size / 4 * (1 if index & (1 << i) else -1) for i in range(3)]
            self.children[index] = OctreeNode(new_center, self.size / 2)
        self.children[index].insert(body, depth)

   
    
    def update_mass_and_center_of_mass(self):
        total_mass = 0
        x, y, z = 0, 0, 0
        for child in self.children:
            if child:
                total_mass += child.total_mass
                x += child.center_of_mass[0] * child.total_mass
                y += child.center_of_mass[1] * child.total_mass
                z += child.center_of_mass[2] * child.total_mass
        if total_mass > 0:
            self.center_of_mass = [x / total_mass, y / total_mass, z / total_mass]
        self.total_mass = total_mass

    def subdivide(self):
        self.is_leaf = False


    def compute_force(self, body, theta=0.5):
        if self.is_leaf and self.body is not None and self.body != body:
            return calculate_force(body, self.body)
        elif not self.is_leaf:
            distance = np.linalg.norm(body.position - self.center_of_mass)
            if distance < 1e-10:  # Un seuil pour éviter la division par zéro
                return np.array([0, 0, 0])
            
            if (self.size / distance) < theta:
                # Le nœud est suffisamment loin pour être traité comme un seul corps
                mock_body = Body(self.center_of_mass, self.total_mass, [0, 0, 0])
                return calculate_force(body, mock_body)
            else:
                # Parcours récursif des enfants pour un calcul plus précis
                force = np.array([0.0, 0.0, 0.0], dtype=float)
                for child in self.children:
                    if child:
                        force += child.compute_force(body, theta)
                return force
        else:
            return np.array([0.0, 0.0, 0.0], dtype=float)




# Fonction pour calculer la force gravitationnelle
def calculate_force(body1, body2):
    distance = np.linalg.norm(body1.position - body2.position)
    if distance < 1e-10:  # Un seuil petit pour éviter la division par zéro
        return np.array([0, 0, 0])

    force_magnitude = G * body1.mass * body2.mass / (distance ** 2)
    force_direction = (body2.position - body1.position) / distance
    return force_magnitude * force_direction

# Fonction pour mettre à jour les corps
def update_bodies(bodies, octree_root, timestep, theta=0.5):
    for body in bodies:
        body.force = octree_root.compute_force(body, theta)
        body.update_velocity(timestep)
        body.update_position(timestep)

# Fonction principale de la simulation
def run_simulation(bodies, total_time, timestep, theta=0.5):
    positions = {body: [] for body in bodies}
    for t in np.arange(0, total_time, timestep):
        root_center = [0, 0, 0]
        root_size = 1e3
        octree_root = OctreeNode(root_center, root_size)

        for body in bodies:
            octree_root.insert(body)

        update_bodies(bodies, octree_root, timestep, theta)

        # Enregistrer les positions pour la visualisation
        for body in bodies:
            positions[body].append(body.position.copy())

    return positions

# Fonction pour créer des corps avec des positions, masses et vitesses initiales aléatoires
def create_random_bodies(num_bodies, position_range, mass_range, velocity_range):
    bodies = []
    for _ in range(num_bodies):
        position = [random.uniform(*position_range) for _ in range(3)]
        mass = random.uniform(*mass_range)  # Obtenez une valeur de masse
        velocity = [random.uniform(*velocity_range) for _ in range(3)]
        bodies.append(Body(position, mass, velocity))  # Passez la masse ici
    return bodies

# Fonction pour visualiser la simulation
def visualize_simulation(positions):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    lines = [ax.plot([], [], [], 'o')[0] for _ in positions]
    data = {body: np.array(positions[body]) for body in positions}

    def update_plot(num, data, lines):
        for line, body_data in zip(lines, data.values()):
            line.set_data(body_data[:num, :2].T)
            line.set_3d_properties(body_data[:num, 2])
        return lines

    anim = FuncAnimation(fig, update_plot, frames=len(next(iter(positions.values()))), fargs=(data, lines), interval=1, blit=True)
    ax.set_xlim([-1e2, 1e2])
    ax.set_ylim([-1e2, 1e2])
    ax.set_zlim([-1e2, 1e2])
    plt.show()

# Exemple d'utilisation
num_bodies = 500
position_range = (-1e2, 1e2)
mass_range = (1e20, 1e25)
velocity_range = (-1e3, 1e3)  # Intervalles pour les vitesses initiales

# Créer des corps
bodies = create_random_bodies(num_bodies, position_range, mass_range, velocity_range)

# Paramètres de la simulation
total_time = 10    # Durée totale de la simulation en secondes
timestep = 0.01    # Pas de temps de la simulation

# Exécuter la simulation
positions = run_simulation(bodies, total_time, timestep)

# Visualiser la simulation
visualize_simulation(positions)
