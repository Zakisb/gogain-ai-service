"""Training tools for LlamaIndex agents.

This module provides tools for managing training programs, including checking
existing programs, retrieving templates, creating custom programs, and accessing
program details.
"""

import logging
from typing import List
from llama_index.core.tools import FunctionTool

from app.tools.training_helpers import (
    check_user_has_training_program,
    get_training_program_templates,
    select_exercises_with_ai,
    create_user_training_program,
    get_user_training_program_details
)

logger = logging.getLogger(__name__)


def check_user_training_program(user_id: str = None) -> str:
    """Check if the user has an active training program.

    Args:
        user_id: User ID (injected programmatically by workflow, not by LLM)

    Returns:
        String indicating whether user has a training program
    """
    try:
        # user_id is injected by workflow, fallback to context if not provided
        if not user_id:
            from app.knowledge.enhanced_assessment_tools import get_user_context
            user_id = get_user_context()

        has_program = check_user_has_training_program(user_id)

        if has_program:
            return "L'utilisateur a un programme d'entraînement actif. Utilisez get_user_training_program pour voir les détails."
        else:
            return "L'utilisateur n'a pas de programme d'entraînement actif. Vous pouvez proposer de créer un nouveau programme."

    except Exception as e:
        logger.error(f"Error checking user training program: {e}")
        return "Impossible de vérifier le programme d'entraînement de l'utilisateur."


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
        # user_id is injected by workflow, fallback to context if not provided
        if not user_id:
            from app.knowledge.enhanced_assessment_tools import get_user_context
            user_id = get_user_context()
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
            return f"✅ Programme créé avec succès: {program_name}\n\nVous pouvez maintenant consulter votre programme avec get_user_training_program."
        else:
            error_msg = result.get("message", "Erreur inconnue")
            return f"❌ Erreur lors de la création du programme: {error_msg}"

    except Exception as e:
        logger.error(f"Error creating training program: {e}", exc_info=True)
        return f"Erreur lors de la création du programme d'entraînement: {str(e)}"


def get_user_training_program(user_id: str = None) -> str:
    """Get detailed information about the user's current training program.

    This includes all training days, exercises, sets, reps, and rest times.

    Args:
        user_id: User ID (injected programmatically by workflow, not by LLM)

    Returns:
        Formatted string with complete program details
    """
    try:
        # user_id is injected by workflow, fallback to context if not provided
        if not user_id:
            from app.knowledge.enhanced_assessment_tools import get_user_context
            user_id = get_user_context()

        program_details = get_user_training_program_details(user_id)

        if not program_details:
            return "Aucun programme d'entraînement trouvé pour cet utilisateur."

        return program_details

    except Exception as e:
        logger.error(f"Error getting user training program: {e}")
        return "Impossible de récupérer le programme d'entraînement de l'utilisateur."


# Factory functions to create LlamaIndex tools

def get_check_training_program_tool() -> FunctionTool:
    """Factory to create the check_user_training_program tool."""
    return FunctionTool.from_defaults(
        fn=check_user_training_program,
        name="check_user_training_program",
        description=(
            "Vérifie si l'utilisateur a un programme d'entraînement actif. "
            "Utilisez cet outil avant de proposer de créer un nouveau programme. "
            "Le user_id est fourni automatiquement par le système."
        )
    )


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


def get_user_training_program_tool() -> FunctionTool:
    """Factory to create the get_user_training_program tool."""
    return FunctionTool.from_defaults(
        fn=get_user_training_program,
        name="get_user_training_program",
        description=(
            "Récupère les détails complets du programme d'entraînement actuel de l'utilisateur, "
            "incluant tous les jours d'entraînement, exercices, séries, répétitions et temps de repos. "
            "Utilisez cet outil lorsque l'utilisateur demande à voir son programme ou ses exercices. "
            "Le user_id est fourni automatiquement par le système."
        )
    )


def get_all_training_tools() -> List[FunctionTool]:
    """Get all training tools as a list.

    Returns:
        List of all training-related FunctionTool instances
    """
    return [
        get_check_training_program_tool(),
        # get_programs_tool(),  # Removed - not needed for AI-generated programs
        get_create_training_program_tool(),
        get_user_training_program_tool(),
    ]
