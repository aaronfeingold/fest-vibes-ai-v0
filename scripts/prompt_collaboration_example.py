#!/usr/bin/env python3
"""
Example script for non-technical collaborators to modify domain context and prompts.

This script provides a simple interface for content writers, marketers, or other
non-technical team members to modify the bot's personality and behavior without
touching code.
"""

import sys
from pathlib import Path

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.llm_client import LLMClient
from src.config.settings import LLMConfig


def print_current_settings(llm_client: LLMClient):
    """Display current domain context and content type instructions."""
    print("=" * 60)
    print("CURRENT DOMAIN CONTEXT")
    print("=" * 60)
    print(llm_client.get_domain_context())
    print()

    print("=" * 60)
    print("CONTENT TYPE INSTRUCTIONS")
    print("=" * 60)

    for content_type in llm_client.list_available_content_types():
        print(f"\n{content_type.upper()}:")
        print(f"  {llm_client.get_content_type_instruction(content_type)}")
    print()


def update_domain_context_interactive(llm_client: LLMClient):
    """Interactive domain context update for non-technical users."""
    print("=" * 60)
    print("UPDATE DOMAIN CONTEXT")
    print("=" * 60)
    print("Current domain context:")
    print(f"  {llm_client.get_domain_context()}")
    print()

    print("Enter new domain context (or press Enter to keep current):")
    new_context = input("> ")

    if new_context.strip():
        llm_client.update_domain_context(new_context.strip())
        print("[SUCCESS] Domain context updated!")
    else:
        print("[INFO] No changes made.")
    print()


def update_content_type_interactive(llm_client: LLMClient):
    """Interactive content type instruction update."""
    print("=" * 60)
    print("UPDATE CONTENT TYPE INSTRUCTIONS")
    print("=" * 60)

    available_types = llm_client.list_available_content_types()
    print("Available content types:")
    for i, content_type in enumerate(available_types, 1):
        print(f"  {i}. {content_type}")
    print()

    try:
        choice = input(f"Select content type to modify (1-{len(available_types)}): ")
        index = int(choice) - 1

        if 0 <= index < len(available_types):
            content_type = available_types[index]

            print(f"\nCurrent instruction for '{content_type}':")
            print(f"  {llm_client.get_content_type_instruction(content_type)}")
            print()

            print("Enter new instruction (or press Enter to keep current):")
            new_instruction = input("> ")

            if new_instruction.strip():
                llm_client.update_content_type_instruction(
                    content_type, new_instruction.strip()
                )
                print(f"[SUCCESS] Instruction for '{content_type}' updated!")
            else:
                print("[INFO] No changes made.")
        else:
            print("[ERROR] Invalid selection.")

    except (ValueError, IndexError):
        print("[ERROR] Invalid input.")
    print()


def test_prompt_generation(llm_client: LLMClient):
    """Test prompt generation with current settings."""
    print("=" * 60)
    print("TEST PROMPT GENERATION")
    print("=" * 60)

    content_types = llm_client.list_available_content_types()
    print("Testing prompt generation for each content type:")
    print()

    for content_type in content_types:
        print(f"--- {content_type.upper()} PROMPT ---")
        test_prompt = llm_client._prepare_prompt(
            base_prompt="Create engaging content about local music events.",
            content_type=content_type,
            context="Tonight's events include jazz at Preservation Hall and bounce at Hi-Ho Lounge.",
        )

        # Show first 200 characters
        preview = test_prompt[:200] + "..." if len(test_prompt) > 200 else test_prompt
        print(preview)
        print()


def main():
    """Main interactive script for prompt collaboration."""
    print("[MUSIC] Fest Vibes NOLA - Prompt Collaboration Tool")
    print("For content writers, marketers, and non-technical collaborators")
    print()

    # Initialize LLM client (no API keys needed for this demo)
    try:
        config = LLMConfig()
        llm_client = LLMClient(config)
    except Exception as e:
        # Create a minimal client for demonstration
        print("[WARNING] Using demo mode (LLM functionality disabled)")
        from src.config.settings import BotConfig

        config = BotConfig().llm
        llm_client = LLMClient(config)

    while True:
        print("What would you like to do?")
        print("1. View current settings")
        print("2. Update domain context")
        print("3. Update content type instructions")
        print("4. Test prompt generation")
        print("5. Exit")

        choice = input("\nSelect option (1-5): ").strip()
        print()

        if choice == "1":
            print_current_settings(llm_client)
        elif choice == "2":
            update_domain_context_interactive(llm_client)
        elif choice == "3":
            update_content_type_interactive(llm_client)
        elif choice == "4":
            test_prompt_generation(llm_client)
        elif choice == "5":
            print("[GOODBYE] Goodbye!")
            break
        else:
            print("[ERROR] Invalid option. Please try again.")
            print()


if __name__ == "__main__":
    main()
