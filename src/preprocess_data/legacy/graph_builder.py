import logging

try:
    from neo4j import GraphDatabase
except ImportError:
    print("Please run: pip install neo4j")
    exit(1)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("graph_builder")

NEO4J_URI = "bolt://localhost:7688"
NEO4J_USER = "neo4j"
NEO4J_PASS = "movierag123"


class MovieGraphBuilder:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def create_constraints(self):
        with self.driver.session() as session:
            session.run(
                "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Character) REQUIRE c.id IS UNIQUE"
            )
            session.run(
                "CREATE CONSTRAINT IF NOT EXISTS FOR (m:Movie) REQUIRE m.id IS UNIQUE"
            )
            session.run(
                "CREATE CONSTRAINT IF NOT EXISTS FOR (s:Scene) REQUIRE s.id IS UNIQUE"
            )
            logger.info("Constraints created.")

    def import_movie_data(self, data: list):
        """
        Takes processed scene data and pushes into Neo4j.
        Expected structure per item:
        {
            "movie_id": "tt123",
            "clip_id": "tt123_clip1",
            "text": "description",
            "metadata": {
                "situation": "chase",
                "characters": ["Batman", "Joker"]
            }
        }
        """
        with self.driver.session() as session:
            for item in data:
                movie_id = item.get("movie_id")
                clip_id = item.get("clip_id")
                desc = item.get("text", "")
                meta = item.get("metadata", {})
                chars = meta.get("characters", [])
                situation = meta.get("situation", "")

                # Create Movie & Scene
                session.run(
                    """
                    MERGE (m:Movie {id: $movie_id})
                    MERGE (s:Scene {id: $clip_id})
                    SET s.description = $desc, s.situation = $situation
                    MERGE (s)-[:PART_OF]->(m)
                    """,
                    movie_id=movie_id,
                    clip_id=clip_id,
                    desc=desc,
                    situation=situation,
                )

                # Create Characters and relate to Scene
                for char in chars:
                    session.run(
                        """
                        MERGE (c:Character {id: $char_id})
                        SET c.name = $char_name
                        WITH c
                        MATCH (s:Scene {id: $clip_id})
                        MERGE (c)-[:ACTS_IN]->(s)
                        """,
                        char_id=f"{movie_id}_{char}",
                        char_name=char,
                        clip_id=clip_id,
                    )

                logger.info(f"Imported scene {clip_id} with {len(chars)} characters.")

    def import_cast_data(self, meta_dir: str):
        """
        Reads meta/*.json files and ingests Cast data into Neo4j:
        (Actor)-[:ACTS_AS]->(Character)-[:IN_MOVIE]->(Movie)
        """
        import json
        from pathlib import Path

        meta_path = Path(meta_dir)
        if not meta_path.exists():
            logger.error(f"Meta directory not found: {meta_path}")
            return

        with self.driver.session() as session:
            # Create Actor constraint
            session.run(
                "CREATE CONSTRAINT IF NOT EXISTS FOR (a:Actor) REQUIRE a.id IS UNIQUE"
            )

            for file_path in meta_path.glob("*.json"):
                movie_id = file_path.stem
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    cast_list = data.get("cast", [])
                    movie_title = data.get("title", movie_id)

                    # Ensure Movie node exists with title
                    session.run(
                        "MERGE (m:Movie {id: $movie_id}) SET m.title = $title",
                        movie_id=movie_id,
                        title=movie_title,
                    )

                    added_count = 0
                    for member in cast_list:
                        actor_id = member.get("id")
                        actor_name = member.get("name")
                        character_name = member.get("character")

                        if not actor_id or not actor_name or not character_name:
                            continue

                        # Clean character name (remove "(as ...)" or extra spaces)
                        character_name = character_name.split("  ")[0].strip()

                        session.run(
                            """
                            MERGE (a:Actor {id: $actor_id})
                            SET a.name = $actor_name
                            
                            MERGE (c:Character {id: $char_id})
                            SET c.name = $char_name
                            
                            MERGE (m:Movie {id: $movie_id})
                            
                            MERGE (a)-[:ACTS_AS]->(c)
                            MERGE (c)-[:IN_MOVIE]->(m)
                            """,
                            actor_id=actor_id,
                            actor_name=actor_name,
                            char_id=f"{movie_id}_{character_name}",
                            char_name=character_name,
                            movie_id=movie_id,
                        )
                        added_count += 1

                    logger.info(
                        f"Imported {added_count} cast members for movie {movie_id} ({movie_title})"
                    )
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {e}")


if __name__ == "__main__":
    builder = MovieGraphBuilder(NEO4J_URI, NEO4J_USER, NEO4J_PASS)
    builder.create_constraints()
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "cast":
        meta_dir = "movie_data_subset_20/meta"
        print(f"Importing Cast data from {meta_dir}...")
        builder.import_cast_data(meta_dir)
    else:
        print(
            "Graph Build Scripts Ready. Start Neo4j via docker-compose up -d and run this script.\n"
            "Run `python graph_builder.py cast` to import Cast (Actor) data."
        )
    builder.close()
