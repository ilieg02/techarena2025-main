import csv
import networkx as nx

def build_history_kg():
    triples = []

    # ================================
    # 1. Base entity sets
    # ================================
    civilizations = [
        "Ancient Egypt", "Ancient Greece", "Roman Empire", "Byzantine Empire",
        "Ottoman Empire", "Mughal Empire", "Maurya Empire", "Gupta Empire",
        "Han Dynasty", "Qing Dynasty", "Ming Dynasty", "Song Dynasty",
        "Aztec Empire", "Inca Empire", "Maya Civilization", "Babylonian Empire",
        "Persian Empire", "Achaemenid Empire", "Sumerian Civilization",
        "Phoenician Civilization"
    ]

    periods = [
        "Bronze Age", "Iron Age", "Classical Antiquity",
        "Middle Ages", "Early Modern Period", "Modern Era"
    ]

    locations = [
        "Nile Valley", "Mesopotamia", "Indus Valley", "Yellow River Valley",
        "Mediterranean Basin", "Anatolia", "Italian Peninsula",
        "Iberian Peninsula", "Balkan Peninsula", "Arabian Peninsula",
        "Mesoamerica", "Andes Mountains", "Central Asia", "East Asia",
        "South Asia", "Western Europe", "Eastern Europe"
    ]

    people = [
        "Cleopatra", "Julius Caesar", "Augustus", "Genghis Khan", "Kublai Khan",
        "Ashoka", "Akbar", "Suleiman the Magnificent", "Alexander the Great",
        "Napoleon Bonaparte", "George Washington", "Abraham Lincoln",
        "Mahatma Gandhi", "Nelson Mandela", "Winston Churchill", "Charlemagne",
        "Cyrus the Great", "Darius the Great", "Ramses II", "Pericles",
        "Hammurabi", "Marcus Aurelius", "Constantine the Great", "Joan of Arc",
        "Tokugawa Ieyasu", "Queen Victoria", "Louis XIV", "Peter the Great",
        "Catherine the Great", "Simon Bolivar", "Mustafa Kemal Ataturk",
        "Ho Chi Minh", "Chiang Kai-shek", "Mao Zedong", "Sun Yat-sen",
        "Franklin D. Roosevelt", "Theodore Roosevelt", "Adolf Hitler",
        "Joseph Stalin", "Leon Trotsky", "John F. Kennedy",
        "Martin Luther King Jr.", "Che Guevara", "Fidel Castro",
        "Imhotep", "Tutankhamun", "Nefertiti", "Hatshepsut"
    ]

    events = [
        "Battle of Waterloo", "French Revolution", "American Revolution",
        "World War I", "World War II", "Fall of Constantinople",
        "Discovery of America", "Industrial Revolution",
        "Fall of the Berlin Wall", "Russian Revolution", "Meiji Restoration",
        "Unification of Germany", "Unification of Italy", "Partition of India",
        "Signing of Magna Carta", "Crusades", "Peloponnesian War",
        "Punic Wars", "Hundred Years' War", "Thirty Years' War",
        "Opium Wars", "Boxer Rebellion", "Taiping Rebellion",
        "Great Depression", "Moon Landing", "Cuban Missile Crisis",
        "Vietnam War", "Korean War", "Gulf War", "Iraq War",
        "Spanish Civil War", "Nine Years' War", "Glorious Revolution",
        "English Civil War", "First Sino-Japanese War",
        "War of the Roses", "American Civil War", "Reign of Terror",
        "Storming of the Bastille"
    ]

    roles = [
        "king", "queen", "emperor", "pharaoh",
        "politician", "general", "revolutionary", "philosopher"
    ]

    # ================================
    # 2. Type triples
    # ================================
    for civ in civilizations:
        triples.append((civ, "rdf:type", "Civilization"))
    for p in periods:
        triples.append((p, "rdf:type", "HistoricalPeriod"))
    for loc in locations:
        triples.append((loc, "rdf:type", "Region"))
    for person in people:
        triples.append((person, "rdf:type", "Person"))
    for e in events:
        triples.append((e, "rdf:type", "Event"))

    # ================================
    # 3. Civilization → period & origin region
    # ================================
    for i, civ in enumerate(civilizations):
        period = periods[i % len(periods)]
        loc = locations[i % len(locations)]
        triples.append((civ, "hasCorePeriod", period))
        triples.append((civ, "originatedIn", loc))

    # Civilization active in multiple periods (synthetic but useful)
    for civ in civilizations:
        for period in periods:
            triples.append((civ, "potentiallyActiveInPeriod", period))

    # Civilization influence regions
    for civ in civilizations:
        for loc in locations:
            triples.append((civ, "influencedRegion", loc))

    # ================================
    # 4. People → civilization, period, region, role
    # ================================
    for i, person in enumerate(people):
        civ = civilizations[i % len(civilizations)]
        period = periods[i % len(periods)]
        loc = locations[i % len(locations)]
        role = roles[i % len(roles)]

        triples.append((person, "associatedWithCivilization", civ))
        triples.append((person, "livedInPeriod", period))
        triples.append((person, "bornInRegion", loc))
        triples.append((person, "hasRole", role))

    # ================================
    # 5. Events → year, region, period
    # ================================
    years = list(range(1200, 2001, 10))  # 1200, 1210, ..., 2000

    for i, e in enumerate(events):
        year = years[i % len(years)]
        loc = locations[i % len(locations)]
        period = periods[i % len(periods)]

        triples.append((e, "occurredInYear", str(year)))
        triples.append((e, "occurredInRegion", loc))
        triples.append((e, "belongsToPeriod", period))

    # ================================
    # 6. People involved in events
    # ================================
    for i, e in enumerate(events):
        p1 = people[i % len(people)]
        p2 = people[(i + 5) % len(people)]
        triples.append((p1, "involvedIn", e))
        triples.append((p2, "involvedIn", e))

    print(f"Total triples generated: {len(triples)}")  # should be > 1000

    return triples


def build_nx_graph(triples):
    """
    Build an in-memory directed graph (DiGraph) from triples.
    """
    G = nx.DiGraph()
    for h, r, t in triples:
        G.add_node(h)
        G.add_node(t)
        G.add_edge(h, t, relation=r)
    return G


def save_triples_csv(triples, path="history_kg_triples.csv"):
    """
    Save triples to CSV: head, relation, tail
    """
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["head", "relation", "tail"])
        for h, r, t in triples:
            writer.writerow([h, r, t])
    print(f"Saved triples to {path}")


if __name__ == "__main__":
    triples = build_history_kg()

    # Show a small sample
    print("Sample triples:")
    for t in triples[:20]:
        print("  ", t)

    # Save to CSV
    save_triples_csv(triples, "history_kg_triples.csv")

    # Build NetworkX graph
    G = build_nx_graph(triples)
    print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
