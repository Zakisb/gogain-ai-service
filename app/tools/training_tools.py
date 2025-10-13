"""Training tools for LlamaIndex agents.

This module provides tools for managing training programs, including checking
existing programs, retrieving templates, creating custom programs, and accessing
program details.
"""

import logging
from typing import List
from llama_index.core.tools import FunctionTool

from app.core.cache import invalidate_user_training_program_cache
from app.core.memory import get_mem0_client
from app.tools.training_helpers import (
    get_training_program_templates,
    select_exercises_with_ai,
    create_user_training_program,
)

logger = logging.getLogger(__name__)


def get_programs() -> str:
    """Get all available training program templates.

    Returns:
        Formatted string with list of available programs
    """
    try:
        programs = get_training_program_templates()

        if not programs:
            return "Aucun programme d'entraînement disponible."

        result = "Programmes d'entraînement disponibles:\n\n"
        for program in programs:
            result += f"ID: {program['id']}\n"
            result += f"Nom: {program['name']}\n"
            result += f"Description: {program.get('description', 'Aucune description')}\n"
            result += f"Durée: {program.get('duration', 'Non spécifiée')} semaines\n"
            result += f"Fréquence: {program.get('frequency', 'Non spécifiée')} jours/semaine\n\n"

        return result.strip()

    except Exception as e:
        logger.error(f"Error getting programs: {e}")
        return "Impossible de récupérer les programmes d'entraînement."


def save_fitness_goal_assessment(
    user_id: str,
    fitness_goal_summary: str
) -> str:
    """Save fitness goal assessment summary to Mem0 for long-term memory.

    This tool is called at the END of the fitness goal assessment workflow to store
    the complete assessment including user goals, preferences, and constraints.

    Args:
        user_id: User ID (injected programmatically by workflow, not by LLM)
        fitness_goal_summary: Complete summary of fitness goal assessment including:
                             - Primary fitness goal
                             - All answers to goal-specific questions
                             - User preferences (status, availability, tracking)
                             - Constraints and special considerations

    Returns:
        Success or error message
    """
    try:
        # user_id is injected by workflow; fail fast if missing
        if not user_id:
            logger.error("save_fitness_goal_assessment called without user_id")
            return "Impossible d'identifier l'utilisateur pour sauvegarder l'évaluation."

        # Get Mem0 client
        mem0 = get_mem0_client()
        if not mem0:
            logger.error("Mem0 client not initialized")
            return "Service de mémoire non disponible. Impossible de sauvegarder l'évaluation."

        # Save to Mem0 with metadata
        result = mem0.add_memory(
            message=fitness_goal_summary,
            user_id=str(user_id),
            metadata={
                "type": "fitness_goal_assessment",
                "category": "training"
            }
        )

        if result.get("success"):
            logger.info(f"✅ Fitness goal assessment saved to Mem0 for user {user_id}")
            return (
                "✅ Évaluation des objectifs fitness sauvegardée avec succès ! "
                "Je vais maintenant utiliser ces informations pour créer ton programme personnalisé."
            )
        else:
            error = result.get("error", "Erreur inconnue")
            logger.error(f"❌ Failed to save fitness goal assessment: {error}")
            return f"❌ Erreur lors de la sauvegarde de l'évaluation: {error}"

    except Exception as e:
        logger.error(f"Error saving fitness goal assessment: {e}", exc_info=True)
        return f"Erreur lors de la sauvegarde de l'évaluation: {str(e)}"


def create_training_program(
    goal: str,
    level: str,
    training_frequency: int,
    available_equipments: List[str],
    zone_target: str = None,
    user_id: str = None
) -> str:
    """Create a personalized AI-generated training program for the user.

    This tool uses AI to select appropriate exercises based on the user's parameters
    and creates a customized training program stored in the database.

    Args:
        goal: Fitness goal - must be one of: "RESTER EN FORME", "REDESSINER SA SILHOUETTE"
        level: Fitness level - must be one of: "beginner", "intermediate", "advanced"
        training_frequency: Number of training days per week (3, 4, or 5)
        available_equipments: List of available equipment (e.g., ["Haltères", "Tapis", "Barre"])
        zone_target: Optional target body zone - one of: "HAUT DU CORPS", "BAS DU CORPS", "CEINTURE ABDOMINALE", "CORPS COMPLET"
                     Required only if goal is "REDESSINER SA SILHOUETTE", ignored for "RESTER EN FORME"
        user_id: User ID (injected programmatically by workflow, not by LLM)

    Returns:
        Success or error message
    """
    try:
        # user_id is injected by workflow; fail fast if missing to avoid ambiguity
        if not user_id:
            logger.error("create_training_program called without user_id")
            return "Impossible d'identifier l'utilisateur pour créer le programme."
        # Validate inputs
        valid_goals = ["RESTER EN FORME", "REDESSINER SA SILHOUETTE"]
        if goal not in valid_goals:
            return f"Objectif invalide. Doit être l'un de: {', '.join(valid_goals)}"

        valid_levels = ["beginner", "intermediate", "advanced"]
        if level not in valid_levels:
            return f"Niveau invalide. Doit être l'un de: {', '.join(valid_levels)}"

        if training_frequency not in [3, 4, 5]:
            return "Fréquence d'entraînement invalide. Doit être 3, 4 ou 5 jours par semaine."

        # Map goal to focus ID
        goal_to_focus_id = {
            "REDESSINER SA SILHOUETTE": 11,
            "RESTER EN FORME": 12,
        }
        focus_id = goal_to_focus_id[goal]

        logger.info(f"Creating training program for user={user_id}, goal={goal}, level={level}, frequency={training_frequency}")

        # Use AI to select exercises and create program structure
        program_data = select_exercises_with_ai(
            goal=goal,
            zone_target=zone_target,
            level=level,
            training_frequency=training_frequency,
            available_equipments=available_equipments
        )

        if not program_data.get("trainingDays"):
            return "Impossible de générer un programme d'entraînement. Veuillez réessayer."

        # Create program in database
        result = create_user_training_program(
            user_id=user_id,
            program_data=program_data,
            focus_id=focus_id,
            level=level
        )

        if result.get("success"):
            program_name = program_data.get("name", "Programme personnalisé")
            invalidate_user_training_program_cache(str(user_id))
            return (
                f"✅ Programme créé avec succès: {program_name}\n\n"
                "Tu peux me demander à tout moment d'afficher ton programme pour voir les détails."
            )
        else:
            error_msg = result.get("message", "Erreur inconnue")
            return f"❌ Erreur lors de la création du programme: {error_msg}"

    except Exception as e:
        logger.error(f"Error creating training program: {e}", exc_info=True)
        return f"Erreur lors de la création du programme d'entraînement: {str(e)}"


# Factory functions to create LlamaIndex tools

def get_programs_tool() -> FunctionTool:
    """Factory to create the get_programs tool."""
    return FunctionTool.from_defaults(
        fn=get_programs,
        name="get_programs",
        description=(
            "Récupère la liste des programmes d'entraînement disponibles (templates). "
            "Chaque programme a un ID, un nom, une description, une durée et une fréquence."
        )
    )


def get_save_fitness_goal_assessment_tool() -> FunctionTool:
    """Factory to create the save_fitness_goal_assessment tool."""
    return FunctionTool.from_defaults(
        fn=save_fitness_goal_assessment,
        name="save_fitness_goal_assessment",
        description=(
            "Sauvegarde le résumé complet de l'évaluation des objectifs fitness dans Mem0 pour mémoire à long terme. "
            "Cet outil doit être appelé à la FIN du workflow d'évaluation des objectifs, une seule fois, "
            "après avoir collecté toutes les informations nécessaires (objectif principal, réponses aux questions "
            "spécifiques, préférences utilisateur, disponibilités, contraintes). Le user_id est fourni automatiquement "
            "par le système. Paramètres requis: fitness_goal_summary (résumé textuel complet de l'évaluation)."
        )
    )


def get_create_training_program_tool() -> FunctionTool:
    """Factory to create the create_training_program tool."""
    return FunctionTool.from_defaults(
        fn=create_training_program,
        name="create_training_program",
        description=(
            "Crée un programme d'entraînement personnalisé basé sur l'IA en sélectionnant des exercices "
            "appropriés selon les paramètres de l'utilisateur: objectif, niveau, fréquence d'entraînement, "
            "équipements disponibles, et zone cible (optionnelle). Le programme est automatiquement sauvegardé "
            "dans la base de données. Le user_id est fourni automatiquement par le système. "
            "Paramètres requis: goal (RESTER EN FORME ou REDESSINER SA SILHOUETTE), "
            "level (beginner/intermediate/advanced), training_frequency (3/4/5), available_equipments (liste), "
            "et zone_target optionnel (HAUT DU CORPS/BAS DU CORPS/CEINTURE ABDOMINALE/CORPS COMPLET)."
        )
    )

def get_all_training_tools() -> List[FunctionTool]:
    """Get all training tools as a list.

    Returns:
        List of all training-related FunctionTool instances
    """
    return [
        # get_programs_tool(),  # Removed - not needed for AI-generated programs
        get_save_fitness_goal_assessment_tool(),
        get_create_training_program_tool(),
    ]
