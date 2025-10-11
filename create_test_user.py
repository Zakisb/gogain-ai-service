"""Create a test user with complete onboarding profile."""

from app.core.database import initialize_pool, execute_insert, execute_query, get_db_connection
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_test_user():
    """Create a test user with complete profile data."""
    initialize_pool(1, 2)

    with get_db_connection() as conn:
        cursor = conn.cursor()

        try:
            # Create user
            logger.info("Creating test user...")
            cursor.execute("""
                INSERT INTO "User"
                (email, "firstName", "lastName", onboarded, "emailVerified", role, "createdAt", "updatedAt", "accountTypeId")
                VALUES (%s, %s, %s, %s, %s, %s, NOW(), NOW(), %s)
                RETURNING id
            """, (
                'test.user@gogain.com',
                'Test',
                'User',
                True,  # onboarded
                True,  # emailVerified
                'user',
                1  # accountTypeId
            ))

            user_id = cursor.fetchone()[0]
            logger.info(f"✅ Created user with ID: {user_id}")

            # Create user profile with complete onboarding data
            logger.info("Creating user profile...")
            cursor.execute("""
                INSERT INTO "UserProfile"
                (
                    "userId",
                    "birthDate",
                    gender,
                    height,
                    weight,
                    goal,
                    "fitnessLevel",
                    "desiredTrainingFrequency",
                    "availableEquipments",
                    "heightUnit",
                    "weightUnit"
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (
                user_id,
                '1990-01-15',  # birthDate (34 years old)
                'MALE',  # gender
                180.0,  # height in cm
                80.0,  # weight in kg
                'REDESSINER SA SILHOUETTE',  # goal
                'intermediate',  # fitnessLevel
                '4',  # desiredTrainingFrequency (4 days per week)
                ['Haltères', 'Tapis', 'Barre'],  # availableEquipments
                'cm',  # heightUnit
                'kg'  # weightUnit
            ))

            profile_id = cursor.fetchone()[0]
            logger.info(f"✅ Created user profile with ID: {profile_id}")

            conn.commit()

            # Verify creation
            logger.info("\nVerifying user creation...")
            cursor.execute("""
                SELECT
                    u.id,
                    u.email,
                    u."firstName",
                    u."lastName",
                    u.onboarded,
                    up."birthDate",
                    up.gender,
                    up.height,
                    up.weight,
                    up.goal,
                    up."fitnessLevel",
                    up."desiredTrainingFrequency",
                    up."availableEquipments"
                FROM "User" u
                LEFT JOIN "UserProfile" up ON u.id = up."userId"
                WHERE u.id = %s
            """, (user_id,))

            user_data = cursor.fetchone()

            if user_data:
                logger.info("\n" + "=" * 80)
                logger.info("USER CREATED SUCCESSFULLY")
                logger.info("=" * 80)
                logger.info(f"ID:              {user_data[0]}")
                logger.info(f"Email:           {user_data[1]}")
                logger.info(f"Name:            {user_data[2]} {user_data[3]}")
                logger.info(f"Onboarded:       {user_data[4]}")
                logger.info(f"\nProfile:")
                logger.info(f"  Birth Date:    {user_data[5]}")
                logger.info(f"  Gender:        {user_data[6]}")
                logger.info(f"  Height:        {user_data[7]} cm")
                logger.info(f"  Weight:        {user_data[8]} kg")
                logger.info(f"  Goal:          {user_data[9]}")
                logger.info(f"  Fitness Level: {user_data[10]}")
                logger.info(f"  Training Freq: {user_data[11]} days/week")
                logger.info(f"  Equipment:     {user_data[12]}")
                logger.info("=" * 80)

            return user_id

        except Exception as e:
            conn.rollback()
            logger.error(f"❌ Error creating user: {e}")
            raise


if __name__ == "__main__":
    try:
        user_id = create_test_user()
        print(f"\n✅ Test user created successfully with ID: {user_id}")
        print(f"\nYou can now test training tools with this user!")
        print(f"Use user_id '{user_id}' in your chat requests.")
    except Exception as e:
        print(f"\n❌ Failed to create test user: {e}")
        exit(1)
