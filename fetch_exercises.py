# fetch_exercises_sqlalchemy.py
import os
import json
from dotenv import load_dotenv
from sqlalchemy import (
    create_engine, Column, Integer, String, Boolean, Text, ForeignKey, DateTime,
    Table
)
from sqlalchemy.orm import sessionmaker, relationship, declarative_base, joinedload, selectinload
from sqlalchemy.sql import func # For default timestamp

# Load environment variables from .env file
load_dotenv()

DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")

DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# --- SQLAlchemy Setup ---
engine = create_engine(DATABASE_URL, echo=False) # Set echo=True for SQL logging
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# --- Define Simplified ORM Models (matching relevant parts of your schema) ---
# Association Tables for Many-to-Many relationships
exercise_tag_association = Table(
    'ExerciseTag', Base.metadata,
    Column('exerciseId', Integer, ForeignKey('Exercise.id'), primary_key=True),
    Column('tagId', Integer, ForeignKey('Tag.id'), primary_key=True)
    # Add other columns if they exist in the join table, though not needed for relationship loading
)

exercise_equipment_association = Table(
    'ExerciseEquipment', Base.metadata,
    Column('exerciseId', Integer, ForeignKey('Exercise.id'), primary_key=True),
    Column('equipmentId', Integer, ForeignKey('Equipment.id'), primary_key=True)
    # Add other columns if they exist in the join table
)

class ExerciseCategory(Base):
    __tablename__ = 'ExerciseCategory'
    id = Column(Integer, primary_key=True)
    value = Column(String, unique=True, nullable=False)
    description = Column(String)
    exercises = relationship("Exercise", back_populates="category")

class Level(Base):
    __tablename__ = 'Level'
    id = Column(Integer, primary_key=True)
    value = Column(String, unique=True, nullable=False)
    description = Column(String)
    exercises = relationship("Exercise", back_populates="level")

class Tag(Base):
    __tablename__ = 'Tag'
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, nullable=False)
    exercises = relationship(
        "Exercise",
        secondary=exercise_tag_association,
        back_populates="tags"
    )

class Equipment(Base):
    __tablename__ = 'Equipment'
    id = Column(Integer, primary_key=True)
    value = Column(String, unique=True, nullable=False)
    description = Column(String)
    exercises = relationship(
        "Exercise",
        secondary=exercise_equipment_association,
        back_populates="equipments"
    )

class Exercise(Base):
    __tablename__ = 'Exercise'

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    duration = Column(Integer, nullable=True)
    deleted = Column(Boolean, default=False)
    categoryId = Column(Integer, ForeignKey('ExerciseCategory.id'), nullable=False, index=True)
    levelId = Column(Integer, ForeignKey('Level.id'), nullable=False, index=True)
    # Basic timestamp example if needed, adjust as per your actual schema
    createdAt = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    category = relationship("ExerciseCategory", back_populates="exercises")
    level = relationship("Level", back_populates="exercises")
    tags = relationship(
        "Tag",
        secondary=exercise_tag_association,
        back_populates="exercises",
        # lazy='selectin' # Alternative loading strategy
    )
    equipments = relationship(
        "Equipment",
        secondary=exercise_equipment_association,
        back_populates="exercises",
        # lazy='selectin' # Alternative loading strategy
    )

    # Add other fields from your Prisma schema if needed for filtering/display,
    # but not strictly required for the formatting task if not used in the text.

# --- Main Script Logic ---

def fetch_and_format_exercises_sqlalchemy():
    """Connects via SQLAlchemy, fetches exercises with relationships, and formats them."""
    db = SessionLocal()
    formatted_exercises = []
    try:
        print("Querying database via SQLAlchemy...")
        # Query exercises, eager loading related category, level, tags, and equipment
        # joinedload is good for one-to-many/many-to-one
        # selectinload is often better for many-to-many (avoids cartesian product)
        exercises_query = db.query(Exercise).options(
            joinedload(Exercise.category),
            joinedload(Exercise.level),
            selectinload(Exercise.tags),
            selectinload(Exercise.equipments)
        ).filter(Exercise.deleted == False)

        exercises = exercises_query.all()
        print(f"Fetched {len(exercises)} exercises.")

        # Process each exercise object
        for exercise in exercises:
            # Access related objects directly via relationships
            category_name = exercise.category.value if exercise.category else "Unknown Category"
            level_name = exercise.level.value if exercise.level else "Unknown Level"
            tags_list = [tag.name for tag in exercise.tags]
            equipment_list = [eq.value for eq in exercise.equipments]

            # Format the descriptive string
            description = exercise.description or 'No description available.'
            tags_str = ', '.join(tags_list) or 'None'
            equipment_str = ', '.join(equipment_list) or 'Bodyweight'
            duration_str = f"{exercise.duration} seconds" if exercise.duration is not None else 'Duration not specified.'

            formatted_text = f"""Exercise Title: {exercise.title}
                Description: {description}
                Category: {category_name}
                Difficulty Level: {level_name}
                Tags: {tags_str}
                Equipment: {equipment_str}
                Duration: {duration_str}"""

            formatted_exercises.append({
                "doc_id": f"exercise-{exercise.id}",     # stable unique key for the vector DB
                "text": formatted_text.strip(),          # natural-language content for embedding
                "metadata": {
                    "exercise_id": exercise.id,          # link back to PostgreSQL
                    "category": category_name,
                    "difficulty_level": level_name,
                    "equipment_list": equipment_list,
                    "tags_list": tags_list,
                    "duration_seconds": exercise.duration
                }
            })

    except Exception as e:
        print(f"An error occurred: {e}")
        # Consider more specific exception handling (e.g., OperationalError)
    finally:
        db.close()
        print("Database session closed.")

    return formatted_exercises

# --- Execution ---
if __name__ == "__main__":
    # You might want to create tables if they don't exist, but typically
    # the schema is managed elsewhere (like Prisma migrations).
    # Base.metadata.create_all(bind=engine) # Uncomment only if needed for setup

    processed_exercises = fetch_and_format_exercises_sqlalchemy()

    if processed_exercises:
        print(f"\nSuccessfully processed {len(processed_exercises)} exercises.")

        # Option 1: Print the first few results for verification
        print("\n--- Sample Formatted Exercises ---")
        # for exercise_data in processed_exercises[:3]: # Print first 3
        #      print(json.dumps(exercise_data, indent=2))
        #      print("-" * 20)

        # Option 2: Save all results to a JSON file
        output_filename = "formatted_exercises_sqlalchemy.json"
        try:
            with open(output_filename, 'w', encoding='utf-8') as f:
                json.dump(processed_exercises, f, indent=2, ensure_ascii=False)
            print(f"\nFormatted exercises saved to {output_filename}")
        except IOError as e:
            print(f"Error saving file: {e}")
    else:
        print("No exercises were processed.")
