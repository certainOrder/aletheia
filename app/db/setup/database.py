"""
Core database setup implementation.
"""
import logging
import time
from typing import List

from ..utils.remote import RemoteExecutor, CommandResult

logger = logging.getLogger(__name__)

# Package management commands
POSTGRESQL_PACKAGES = [
    "postgresql",
    "postgresql-contrib",
    "postgresql-15",
    "postgresql-client-15"
]

class DatabaseSetup:
    """Main orchestrator for database setup"""
    def __init__(self, host: str, nuclear: bool = False):
        self.host = host
        self.nuclear = nuclear
        self.results = []
        self.executor = RemoteExecutor(host)
        
    def setup(self) -> bool:
        """Runs full setup and returns overall success"""
        results = []
        
        if self.nuclear:
            results.extend([
                self.stop_postgres(),       # bool: service stopped
                self.remove_postgres(),     # bool: packages removed
                self.purge_data()          # bool: configs/data purged
            ])
            
        results.extend([
            self.install_packages(),    # bool: packages installed
            self.create_databases(),    # bool: databases exist
            self.deploy_schemas()       # bool: schemas loaded
        ])
        
        self.log_results(results)
        return all(results)
    
    def validate_postgres_stopped(self) -> bool:
        """Check if PostgreSQL service is stopped"""
        result = self.executor.run_command(
            "/usr/bin/systemctl is-active postgresql",
            use_sudo=True
        )
        return result.stdout.strip() in ["inactive", "unknown"]

    def stop_postgres(self) -> bool:
        """Stop PostgreSQL service"""
        logger.info("Stopping PostgreSQL service...")
        # Perform action
        result = self.executor.run_command(
            "/usr/bin/systemctl stop postgresql",
            use_sudo=True
        )
        # Validate result
        success = self.validate_postgres_stopped()
        # Log result
        logger.info(f"PostgreSQL service stop {'succeeded' if success else 'failed'}")
        return success
        
    def remove_postgres(self) -> bool:
        """Remove PostgreSQL packages"""
        logger.info("Removing PostgreSQL packages...")
        # First get list of installed postgres packages with sudo
        list_cmd = "dpkg -l | grep postgresql | awk '{print $2}'"
        result = self.executor.run_command(list_cmd, use_sudo=True)
        
        if result.stdout.strip():
            # Remove found packages
            remove_cmd = f"DEBIAN_FRONTEND=noninteractive apt-get remove --purge -y {result.stdout.replace(chr(10), ' ')}"
            result = self.executor.run_command(remove_cmd, use_sudo=True)
            if not result.success:
                return False
                
        # Clean up any remaining configs
        cleanup_cmd = "DEBIAN_FRONTEND=noninteractive apt-get autoremove -y && apt-get autoclean -y"
        result = self.executor.run_command(cleanup_cmd, use_sudo=True)
        return True  # Continue even if cleanup fails
        
    def purge_data(self) -> bool:
        """Purge PostgreSQL data and configs"""
        logger.info("Purging PostgreSQL data and configurations...")
        commands = [
            "rm -rf /etc/postgresql/",
            "rm -rf /var/lib/postgresql/",
            "rm -rf /var/log/postgresql/",
            "userdel -r postgres",  # -r removes home directory too
            "groupdel postgres"
        ]
        
        results = self.executor.run_commands(commands, use_sudo=True)
        # It's okay if some commands fail (e.g., if files already don't exist)
        # We only care that the commands were attempted
        return True
        
    def install_packages(self) -> bool:
        """Install required PostgreSQL packages"""
        logger.info("Installing PostgreSQL packages...")
        
        # Update package lists
        result = self.executor.run_command(
            "DEBIAN_FRONTEND=noninteractive apt-get update",
            use_sudo=True
        )
        if not result.success:
            return False
            
        # Install packages
        packages = " ".join(POSTGRESQL_PACKAGES)
        result = self.executor.run_command(
            f"DEBIAN_FRONTEND=noninteractive apt-get install -y {packages}",
            use_sudo=True
        )
        return result.success
        
    def start_postgres(self) -> bool:
        """Start PostgreSQL service"""
        logger.info("Starting PostgreSQL service...")
        result = self.executor.run_command(
            "/usr/bin/systemctl start postgresql",
            use_sudo=True
        )
        if not result.success:
            return False
            
        # Wait for PostgreSQL to be ready
        for _ in range(5):  # Try 5 times
            result = self.executor.run_command(
                "pg_isready",
                use_sudo=True
            )
            if result.success:
                return True
            time.sleep(1)
        return False

    def validate_users_exist(self) -> bool:
        """Check if required users exist"""
        for user in ['hearthminds', 'logos', 'aletheia']:
            check_cmd = f"sudo -u postgres psql -tc \"SELECT 1 FROM pg_roles WHERE rolname = '{user}'\""
            result = self.executor.run_command(check_cmd, use_sudo=True)
            if not (result.success and "1" in result.stdout):
                logger.error(f"User {user} does not exist")
                return False
        return True

    def validate_databases_exist(self) -> bool:
        """Check if all required databases exist"""
        for db in ['hearthminds', 'logos', 'aletheia']:
            check_cmd = f"sudo -u postgres psql -tc \"SELECT 1 FROM pg_database WHERE datname = '{db}'\""
            result = self.executor.run_command(check_cmd, use_sudo=True)
            if not (result.success and "1" in result.stdout):
                logger.error(f"Database {db} does not exist")
                return False
        return True

    def create_databases(self) -> bool:
        """Create all required databases"""
        logger.info("Creating databases and users...")
        
        # Make sure PostgreSQL is running first
        if not self.start_postgres():
            logger.error("Failed to start PostgreSQL")
            return False
            
        # Initial SQL setup - create users and databases
        setup_sql = """
        -- Create admin user
        CREATE USER hearthminds WITH PASSWORD 'hearthminds' SUPERUSER;
        
        -- Create AI users
        CREATE USER logos WITH PASSWORD 'logos';
        CREATE USER aletheia WITH PASSWORD 'aletheia';
        
        -- Create databases with proper encoding and template
        CREATE DATABASE hearthminds WITH OWNER = hearthminds ENCODING = 'UTF8' TEMPLATE = template0;
        CREATE DATABASE logos WITH OWNER = logos ENCODING = 'UTF8' TEMPLATE = template0;
        CREATE DATABASE aletheia WITH OWNER = aletheia ENCODING = 'UTF8' TEMPLATE = template0;
        """
        
        # Perform action - with error stopping enabled
        result = self.executor.run_command(
            f"""sudo -i -u postgres psql -v ON_ERROR_STOP=1 <<EOF
{setup_sql}
EOF""",
            use_sudo=True
        )
        
        if not result.success:
            logger.error(f"SQL execution failed: {result.stderr}")
            return False
        
        # Check users were created first
        if not self.validate_users_exist():
            return False
            
        # Then validate databases exist
        success = self.validate_databases_exist()
        
        # Log result
        logger.info(f"Database creation {'succeeded' if success else 'failed'}")
        return success
            
        # Verify each database exists before proceeding
        for db in ['hearthminds', 'logos', 'aletheia']:
            for _ in range(5):  # Try up to 5 times
                check_cmd = f"sudo -u postgres psql -tc \"SELECT 1 FROM pg_database WHERE datname = '{db}'\""
                result = self.executor.run_command(check_cmd, use_sudo=True)
                if result.success and "1" in result.stdout:
                    break
                logger.info(f"Waiting for database {db} to be ready...")
                time.sleep(1)
            else:
                logger.error(f"Database {db} was not created successfully")
                return False
            
        # Enable vector extension in each database
        for db in ['hearthminds', 'logos', 'aletheia']:
            result = self.executor.run_command(
                f"""sudo -u postgres bash -c 'psql -d {db} -c "CREATE EXTENSION IF NOT EXISTS vector;"'""",
                use_sudo=True
            )
            if not result.success:
                logger.error(f"Failed to enable vector extension in {db}")
                return False
                
        # Set up cross-database access
        access_sql = """
        GRANT CONNECT ON DATABASE hearthminds TO logos, aletheia;
        GRANT CONNECT ON DATABASE logos TO hearthminds;
        GRANT CONNECT ON DATABASE aletheia TO hearthminds;
        """
        
        result = self.executor.run_command(
            f"echo {access_sql!r} | sudo -u postgres psql",
            use_sudo=True
        )
        return result.success
        
    def validate_schema(self, db: str, table_name: str) -> bool:
        """Check if a schema has been properly deployed by checking for a key table"""
        check_cmd = f"""sudo -i -u postgres psql -d {db} -tc "SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = '{table_name}')" """
        result = self.executor.run_command(check_cmd, use_sudo=True)
        if not (result.success and "t" in result.stdout.lower()):
            logger.error(f"Schema validation failed for {db}: {table_name} table not found")
            return False
        return True

    def deploy_schema_to_db(self, db: str, schema_path: str) -> bool:
        """Deploy a schema to a specific database"""
        logger.info(f"Deploying schema to {db}...")
        
        # Read the schema file
        try:
            with open(schema_path, 'r') as f:
                schema_sql = f.read().strip()
        except Exception as e:
            logger.error(f"Failed to read schema file {schema_path}: {e}")
            return False

        # Deploy the schema with proper environment
        result = self.executor.run_command(
            f"""cat <<'EOF' | sudo -i -u postgres psql -v ON_ERROR_STOP=1 -d {db}
{schema_sql}
EOF""",
            use_sudo=True
        )

        if not result.success:
            logger.error(f"Failed to deploy schema to {db}: {result.stderr}")
            return False
            
        return True

    def deploy_schemas(self) -> bool:
        """Deploy SQL schemas to databases"""
        logger.info("Deploying database schemas...")
        
        # Map of database to (schema file, validation table)
        schema_configs = {
            'hearthminds': ('/home/chapinad/projects/openai_pgvector_api/app/db/schema/hm_schema.sql', 'eng_patterns'),
            'logos': ('/home/chapinad/projects/openai_pgvector_api/app/db/schema/pp_schema.sql', 'user_profiles'),
            'aletheia': ('/home/chapinad/projects/openai_pgvector_api/app/db/schema/pp_schema.sql', 'user_profiles')
        }
        
        # Deploy each schema
        for db, (schema_file, check_table) in schema_configs.items():
            # Perform action
            if not self.deploy_schema_to_db(db, schema_file):
                return False
                
            # Validate result
            if not self.validate_schema(db, check_table):
                return False
                
            # Log result
            logger.info(f"Schema deployment to {db} succeeded")
            
        return True
        
    def log_results(self, results: List[bool]) -> None:
        """Log operation results"""
        self.results = results
        for i, result in enumerate(results):
            logger.info(f"Operation {i}: {'Success' if result else 'Failed'}")