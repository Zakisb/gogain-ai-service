"""Helper functions for training program management and AI-based exercise selection.

This module contains database query functions and AI generation logic for creating
personalized training programs.
"""

import logging
import json
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from openai import OpenAI

from app.core.database import execute_query, execute_insert, run_transaction
from app.core.config import OPENAI_API_KEY

logger = logging.getLogger(__name__)

# Initialize OpenAI client
openai_client = OpenAI(api_key=OPENAI_API_KEY)


# System prompt for exercise selection
EXERCISE_SELECTION_PROMPT = """Vous êtes un expert en fitness AI chargé de créer des programmes d'entraînement personnalisés pour les utilisateurs en fonction de leurs objectifs, zones cibles (si spécifiées), niveaux de forme physique et équipements disponibles.

IMPORTANT - SÉLECTION DES EXERCICES BASÉE SUR L'ÉQUIPEMENT :
- Vous recevrez une liste d'exercices avec leurs titres et descriptions détaillées
- Vous recevrez également la liste des équipements disponibles pour l'utilisateur (availableEquipments)
- Vous devez analyser intelligemment le titre et la description de chaque exercice pour déterminer s'il est réalisable avec les équipements disponibles
- Règles pour la sélection des exercices :
  1. Si un exercice mentionne un équipement spécifique dans son titre ou sa description, vérifiez qu'il est disponible dans availableEquipments
  2. Les exercices avec poids du corps (pompes, squats, etc.) sont toujours disponibles
  3. Si un exercice mentionne "haltères" ou "bandes de résistance", vérifiez la présence de ces équipements
  4. Pour les exercices de gym (machines, bancs, etc.), vérifiez si "gym_access" est dans availableEquipments
  5. En cas de doute sur l'équipement requis, privilégiez les exercices alternatifs dont les exigences d'équipement sont claires

Considérez les paramètres suivants :

1. Objectif :
   - REDESSINER SA SILHOUETTE
   - RESTER EN FORME

2. Zone cible (principalement pour REDESSINER SA SILHOUETTE) :
   - HAUT DU CORPS
   - BAS DU CORPS
   - CEINTURE ABDOMINALE
   - NON SPÉCIFIÉ

3. Niveau :
   - DÉBUTANT
   - INTERMÉDIAIRE
   - CONFIRMÉ

4. Fréquence d'entraînement :
   - Créez exactement le nombre de jours spécifié dans trainingFrequency (3, 4 ou 5 jours)

5. Équipements disponibles :
   - Utilisez UNIQUEMENT les exercices réalisables avec les équipements listés dans availableEquipments
   - Analysez attentivement le titre et la description de chaque exercice pour vérifier la compatibilité
   - En cas de doute, privilégiez la sécurité et choisissez un exercice alternatif

Directives pour la création du programme :

1. Créez un programme d'entraînement avec le nombre exact de jours spécifié dans trainingFrequency.

2. Pour chaque jour d'entraînement :
   - Choisissez un focus approprié (par exemple, haut du corps, bas du corps, full body, cardio, etc.)
   - Sélectionnez 5 à 8 exercices appropriés pour ce focus
   - Vérifiez que chaque exercice sélectionné peut être réalisé avec les équipements disponibles
   - Donnez un nom descriptif et motivant à la séance du jour

3. Assurez-vous que le programme global est équilibré et correspond à l'objectif de l'utilisateur :
   - Pour "REDESSINER SA SILHOUETTE" : Concentrez-vous sur le renforcement musculaire et la définition
   - Pour "RESTER EN FORME" : Créez un mélange équilibré de cardio, force et flexibilité

4. Adaptez la difficulté et la variété des exercices au niveau de l'utilisateur

Format de réponse :
Fournissez votre réponse au format JSON suivant :
{
  "name": string,
  "description": string,
  "trainingDays": [
    {
      "dayNumber": number,
      "name": string,
      "exerciseIds": [number, number, ...]
    }
  ]
}

- "name" est une chaîne courte (max 50 caractères) décrivant le programme complet.
- "description" explique brièvement comment le programme répond à l'objectif et au niveau de l'utilisateur.
- "trainingDays" contient exactement trainingFrequency éléments.
  - "dayNumber" commence à 1 et représente le jour d'entraînement.
  - "name" décrit la séance.
  - "exerciseIds" liste les ID des exercices choisis (compatibles avec l'équipement disponible).

Assurez-vous que la réponse est strictement au format JSON ci-dessus."""


def check_user_has_training_program(user_id: str) -> bool:
    """Check if user has an active training program.

    Args:
        user_id: User ID

    Returns:
        True if user has a training program, False otherwise
    """
    try:
        query = """
            SELECT id
            FROM "UserTrainingProgram"
            WHERE "userId" = %s
            ORDER BY "createdAt" DESC
            LIMIT 1
        """
        result = execute_query(query, (int(user_id),), fetch_one=True)
        return result is not None
    except Exception as e:
        logger.error(f"Error checking user training program: {e}")
        return False


def get_training_program_templates() -> List[Dict[str, Any]]:
    """Get all available training program templates.

    Returns:
        List of program templates with id, name, description
    """
    try:
        query = """
            SELECT id, name, description, duration, frequency, "focusId"
            FROM "TrainingProgramTemplate"
            ORDER BY name
        """
        results = execute_query(query)
        return results or []
    except Exception as e:
        logger.error(f"Error fetching training program templates: {e}")
        return []


def get_exercises_by_level(level: str) -> List[Dict[str, Any]]:
    """Get exercises matching the specified fitness level.

    Args:
        level: Fitness level (beginner, intermediate, advanced)

    Returns:
        List of exercises with details
    """
    try:
        # Map level strings to database values
        level_map = {
            "beginner": "beginner",
            "intermediate": "intermediate",
            "advanced": "advanced"
        }
        db_level = level_map.get(level.lower(), "intermediate")

        query = """
            SELECT
                e.id,
                e.title as name,
                e.description,
                l.value as level,
                c.value as category,
                ARRAY_AGG(DISTINCT t.name) as tags,
                ARRAY_AGG(DISTINCT eq.value) as equipments
            FROM "Exercise" e
            LEFT JOIN "Level" l ON e."levelId" = l.id
            LEFT JOIN "ExerciseCategory" c ON e."categoryId" = c.id
            LEFT JOIN "ExerciseTag" et ON e.id = et."exerciseId"
            LEFT JOIN "Tag" t ON et."tagId" = t.id
            LEFT JOIN "ExerciseEquipment" ee ON e.id = ee."exerciseId"
            LEFT JOIN "Equipment" eq ON ee."equipmentId" = eq.id
            WHERE l.value = %s
            GROUP BY e.id, e.title, e.description, l.value, c.value
            ORDER BY e.title
        """
        results = execute_query(query, (db_level,))
        return results or []
    except Exception as e:
        logger.error(f"Error fetching exercises by level: {e}")
        return []


def select_exercises_with_ai(
    goal: str,
    zone_target: Optional[str],
    level: str,
    training_frequency: int,
    available_equipments: List[str]
) -> Dict[str, Any]:
    """Use OpenAI to select appropriate exercises and create a training program.

    Args:
        goal: User's fitness goal (e.g., "RESTER EN FORME", "REDESSINER SA SILHOUETTE")
        zone_target: Target body zone (optional, e.g., "HAUT DU CORPS", "BAS DU CORPS")
        level: Fitness level (beginner, intermediate, advanced)
        training_frequency: Number of training days per week (3, 4, or 5)
        available_equipments: List of available equipment names

    Returns:
        Dict with program structure (name, description, trainingDays)
    """
    try:
        # Get exercises from database
        exercises = get_exercises_by_level(level)

        if not exercises:
            logger.warning(f"No exercises found for level: {level}")
            return {
                "name": "Programme par défaut",
                "description": "Aucun exercice trouvé pour ce niveau",
                "trainingDays": []
            }

        # Format exercises for AI
        formatted_exercises = []
        for ex in exercises:
            formatted_exercises.append({
                "id": ex["id"],
                "name": ex["name"],
                "description": ex.get("description", ""),
                "level": ex.get("level", ""),
                "category": ex.get("category", ""),
                "tags": ex.get("tags", []) or [],
                "equipments": ex.get("equipments", []) or [],
            })

        # Build user parameters
        params = {
            "goal": goal,
            "zoneTarget": zone_target or "NON SPÉCIFIÉ",
            "level": level,
            "trainingFrequency": training_frequency,
            "availableEquipments": available_equipments
        }

        # Create messages for OpenAI
        messages = [
            {"role": "system", "content": EXERCISE_SELECTION_PROMPT},
            {"role": "user", "content": f"User Parameters:\n{json.dumps(params, ensure_ascii=False, indent=2)}"},
            {"role": "user", "content": f"Available Exercises:\n{json.dumps(formatted_exercises, ensure_ascii=False, indent=2)}"}
        ]

        # Call OpenAI
        logger.info(f"Calling OpenAI for exercise selection (level={level}, frequency={training_frequency})")
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.7,
            max_tokens=4096
        )

        # Parse and normalize response
        raw_content = response.choices[0].message.content
        result = json.loads(raw_content)
        training_days = result.get("trainingDays") or []

        for day in training_days:
            exercise_ids = day.get("exerciseIds") or []
            exercises = day.get("exercises") or []

            # Normalize exercise IDs to integers when possible
            normalized_ids: List[Any] = []
            for eid in exercise_ids:
                try:
                    normalized_ids.append(int(eid))
                except (TypeError, ValueError):
                    normalized_ids.append(eid)
            if normalized_ids:
                day["exerciseIds"] = normalized_ids

            if exercise_ids and not exercises:
                # Expand compact exerciseIds list into detailed exercise objects for compatibility
                expanded = []
                for index, eid in enumerate(normalized_ids, start=1):
                    expanded.append({
                        "exerciseId": eid,
                        "setsPlanned": 3,
                        "repsPerSet": 10,
                        "restTimeBetweenSets": 45,
                        "orderInDay": index,
                    })
                day["exercises"] = expanded
            elif exercises:
                for index, exercise in enumerate(exercises, start=1):
                    exercise.setdefault("orderInDay", index)

        logger.info(f"✅ AI generated program: {result.get('name', 'Unknown')}")

        return result

    except Exception as e:
        logger.error(f"Error selecting exercises with AI: {e}", exc_info=True)
        return {
            "name": "Programme d'entraînement personnalisé",
            "description": "Erreur lors de la génération du programme",
            "trainingDays": []
        }


def create_user_training_program(
    user_id: str,
    program_data: Dict[str, Any],
    focus_id: int,
    level: str
) -> Dict[str, Any]:
    """Create a new training program for the user in the database.

    Args:
        user_id: User ID
        program_data: Program data from AI (name, description, trainingDays)
        focus_id: Training focus ID (goal)
        level: Fitness level

    Returns:
        Dict with success status and message
    """
    try:
        start_date = datetime.now().date()
        duration_weeks = 8  # Default 8 weeks
        end_date = start_date + timedelta(weeks=duration_weeks)

        training_days = program_data.get("trainingDays", [])
        frequency = len(training_days)

        async def _create(tx) -> Dict[str, Any]:
            # Ensure only one active program by deactivating existing ones before creating the new program
            await tx.execute(
                """
                UPDATE "UserTrainingProgram"
                SET status = 'INACTIVE'
                WHERE "userId" = %s AND status = 'ACTIVE'
                """,
                (int(user_id),),
            )

            program_id = await tx.insert(
                """
                INSERT INTO "UserTrainingProgram"
                (name, description, "userId", "focusId", duration, frequency, "startDate", "endDate", status)
                VALUES (%s, %s, %s, %s, %s, %s, %s::timestamp, %s::timestamp, %s::"TrainingProgramStatus")
                RETURNING id
                """,
                (
                    program_data.get("name", "Programme personnalisé"),
                    program_data.get("description", ""),
                    int(user_id),
                    focus_id,
                    duration_weeks,
                    frequency,
                    start_date.isoformat(),  # Convert date to string
                    end_date.isoformat(),    # Convert date to string
                    "ACTIVE",
                ),
            )

            if program_id is None:
                raise RuntimeError("Failed to create UserTrainingProgram record")

            logger.info(f"✅ Created UserTrainingProgram with id={program_id}")

            current_date = start_date
            for day_data in training_days:
                day_number = day_data.get("dayNumber", 1)
                day_name = day_data.get("name", f"Jour {day_number}")

                training_day_id = await tx.insert(
                    """
                    INSERT INTO "UserTrainingDay"
                    ("userProgramId", "dayNumber", name, date, status)
                    VALUES (%s, %s, %s, %s::timestamp, %s::"TrainingStatus")
                    RETURNING id
                    """,
                    (
                        program_id,
                        day_number,
                        day_name,
                        current_date.isoformat(),  # Convert date to string
                        "PLANNED",
                    ),
                )

                if training_day_id is None:
                    raise RuntimeError("Failed to create UserTrainingDay record")

                for exercise_data in day_data.get("exercises", []):
                    await tx.execute(
                        """
                        INSERT INTO "UserProgramExercise"
                        ("userTrainingDayId", "exerciseId", "setsPlanned", "repsPerSet", "restTimeBetweenSets", "orderInDay")
                        VALUES (%s, %s, %s, %s, %s, %s)
                        """,
                        (
                            training_day_id,
                            exercise_data.get("exerciseId"),
                            exercise_data.get("setsPlanned", 3),
                            exercise_data.get("repsPerSet", 10),
                            exercise_data.get("restTimeBetweenSets", 60),
                            0,
                        ),
                    )

                current_date += timedelta(days=1)

            logger.info(
                "✅ Successfully created training program with %d days", len(training_days)
            )

            return {
                "success": True,
                "message": "Programme créé avec succès",
                "programId": program_id,
            }

        return run_transaction(_create)

    except Exception as e:
        logger.error(f"Error creating user training program: {e}", exc_info=True)
        return {
            "success": False,
            "message": f"Erreur lors de la création du programme: {str(e)}"
        }


def get_user_training_program_details(user_id: str) -> Optional[str]:
    """Get detailed information about user's current training program.

    Args:
        user_id: User ID

    Returns:
        Formatted string with program details, or None if no program exists
    """
    try:
        # Get program with all related data
        query = """
            SELECT
                utp.id,
                utp.name,
                utp.description,
                utp.duration,
                utp.frequency,
                utp."startDate",
                utp."endDate",
                tpf.name as focus_name,
                td.id as day_id,
                td."dayNumber",
                td.name as day_name,
                td.date as day_date,
                td.status as day_status,
                tde.id as exercise_id,
                e.title as exercise_name,
                e.description as exercise_description,
                tde."setsPlanned",
                tde."repsPerSet",
                tde."restTimeBetweenSets"
            FROM "UserTrainingProgram" utp
            LEFT JOIN "TrainingProgramFocus" tpf ON utp."focusId" = tpf.id
            LEFT JOIN "UserTrainingDay" td ON utp.id = td."userProgramId"
            LEFT JOIN "UserProgramExercise" tde ON td.id = tde."userTrainingDayId"
            LEFT JOIN "Exercise" e ON tde."exerciseId" = e.id
            WHERE utp."userId" = %s
            ORDER BY utp."createdAt" DESC, td."dayNumber", tde."orderInDay"
            LIMIT 1000
        """

        results = execute_query(query, (int(user_id),))

        if not results:
            return None

        # Group results by program
        program_info = {
            "name": results[0]["name"],
            "description": results[0]["description"],
            "focus": results[0].get("focus_name", "Non spécifié"),
            "duration": f"{results[0]['duration']} semaines",
            "frequency": f"{results[0]['frequency']} jours par semaine",
            "startDate": results[0]["startdate"].strftime("%Y-%m-%d") if results[0].get("startdate") else "Non spécifié",
            "endDate": results[0]["enddate"].strftime("%Y-%m-%d") if results[0].get("enddate") else "Non spécifié",
            "trainingDays": {}
        }

        # Group by training day
        for row in results:
            day_id = row.get("day_id")
            if not day_id or day_id in program_info["trainingDays"]:
                continue

            day_number = row.get("daynumber")
            if day_number not in program_info["trainingDays"]:
                program_info["trainingDays"][day_number] = {
                    "dayNumber": day_number,
                    "name": row.get("day_name", f"Jour {day_number}"),
                    "date": row.get("day_date").strftime("%Y-%m-%d") if row.get("day_date") else "Non spécifié",
                    "status": row.get("day_status", "NOT_STARTED"),
                    "exercises": []
                }

        # Add exercises to days
        for row in results:
            day_number = row.get("daynumber")
            if day_number and row.get("exercise_id"):
                program_info["trainingDays"][day_number]["exercises"].append({
                    "name": row.get("exercise_name", "Exercice inconnu"),
                    "description": row.get("exercise_description", ""),
                    "sets": row.get("setsplanned", 3),
                    "reps": row.get("repsperset", 10),
                    "restTime": f"{row.get('resttimebetweensets', 60)} secondes"
                })

        # Format as text context for AI
        formatted_text = f"""
Le programme d'entraînement actuel de l'utilisateur:
Nom: {program_info['name']}
Description: {program_info['description']}
Objectif: {program_info['focus']}
Durée: {program_info['duration']}
Fréquence: {program_info['frequency']}
Date de début: {program_info['startDate']}
Date de fin: {program_info['endDate']}

Le programme comprend {len(program_info['trainingDays'])} jours d'entraînement:
"""

        for day in sorted(program_info["trainingDays"].values(), key=lambda x: x["dayNumber"]):
            formatted_text += f"\nJour {day['dayNumber']} ({day['name']}) - {day['date']}:\n"
            for ex in day["exercises"]:
                formatted_text += f"  - {ex['name']}: {ex['sets']} séries de {ex['reps']} répétitions, {ex['restTime']} de repos\n"

        return formatted_text.strip()

    except Exception as e:
        logger.error(f"Error getting user training program details: {e}", exc_info=True)
        return None
