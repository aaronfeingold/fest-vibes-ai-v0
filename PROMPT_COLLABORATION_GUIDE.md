# Prompt Collaboration Guide

_For content writers, marketers, and non-technical collaborators_

This guide explains how to modify the bot's personality and behavior without touching code.

## What You Can Change

### Domain Context

The **domain context** defines what the bot is and who it's for. This is the core personality.

**Current**: _"You are creating content for a Twitter bot focused on New Orleans culture, music scene, and GenZ trends. The content should be authentic, engaging, and reflect the vibrant culture of New Orleans."_

### Content Type Instructions

These control how the bot writes different types of content:

- **Posts**: Original tweets and social media content
- **Comments**: Replies to other people's tweets
- **Repost Comments**: Text added when sharing someone else's content

## How to Make Changes

### Method 1: Interactive Script (Recommended)

```bash
python scripts/prompt_collaboration_example.py
```

This launches a friendly menu where you can:

- View current settings
- Update domain context
- Modify content type instructions
- Test how changes affect generated prompts

### Method 2: Direct Code Changes

Edit the file: `src/prompts/base_prompts.py`

Look for these sections:

```python
DOMAIN_CONTEXT = (
    "Your new domain context here..."
)

CONTENT_TYPE_INSTRUCTIONS = {
    "post": "Your new post instructions...",
    "comment": "Your new comment instructions...",
}
```

## Writing Tips

### Domain Context

- Keep it conversational and clear
- Define the bot's purpose and audience
- Mention key themes (New Orleans, music, GenZ)
- Set the overall tone and personality

**Good examples:**

- "You create content for music lovers in New Orleans..."
- "You're a friendly guide to NOLA's underground scene..."
- "You help GenZ discover authentic local culture..."

### Content Instructions

- Be specific about format (character limits, style)
- Include do's and don'ts
- Mention key topics to focus on
- Set engagement expectations

**Good examples:**

- "Create tweets under 280 characters that feel like they're from a local friend"
- "Reply with genuine enthusiasm, ask questions, suggest local spots"
- "Share content with personal insight, avoid just repeating what's already said"

## Testing Your Changes

After making changes:

1. **Use the test function** in the collaboration script
2. **Check multiple content types** - make sure posts, comments, and reposts all sound right
3. **Review the full prompt** - see how your changes combine with other instructions
4. **Test with real scenarios** - try different topics and contexts

## How Changes Take Effect

- Changes are **immediate** when using the interactive script
- If editing files directly, **restart the bot** for changes to apply
- Changes affect **all future content generation**
- Previous content is not changed

## Best Practices

### Domain Context

- Clear, concise personality definition
- Specific to your brand and audience
- Sets expectations for tone and style
- Don't make it too long or complicated
- Avoid technical jargon

### Content Instructions

- Include practical constraints (length, format)
- Give positive examples of what to do
- Be specific about your brand voice
- Don't contradict the domain context
- Avoid overly restrictive rules that limit creativity

## File Structure

```
src/prompts/
├── base_prompts.py      # Main settings (edit here)
├── agent_prompts.py     # Specialized prompts
└── rag_prompts.py      # Event-specific prompts

scripts/
└── prompt_collaboration_example.py  # Interactive tool
```

## Collaboration Workflow

1. **Content team** updates domain context and instructions using the interactive tool
2. **Test changes** with the built-in preview function
3. **Coordinate with developers** if you need new content types or features
4. **Document changes** for future reference

## Troubleshooting

**My changes aren't showing up:**

- Make sure you saved the file or completed the interactive script
- Restart the bot if editing files directly
- Check that you modified the right content type

**Bot responses don't match my instructions:**

- Review the complete generated prompt using the test function
- Make sure your instructions are clear and specific
- Check that different instruction types don't conflict

**Need help?**

- Use the interactive script's test function to debug
- Ask developers to add new content types if needed
- Review example prompts for inspiration

## Examples

### Making the Bot More Casual

**Before**: "Create professional social media content..."
**After**: "Write like you're texting a friend about cool stuff happening in NOLA..."

### Adding Focus on Food

**Before**: "Focus on music and culture..."  
**After**: "Focus on music, culture, and the amazing food scene..."

### Encouraging More Interaction

**Before**: "Create engaging tweets..."
**After**: "Create tweets that spark conversation - ask questions, share hot takes, get people talking..."

---

_Remember: Good prompts lead to good content. Take time to craft instructions that capture your brand voice and goals!_
