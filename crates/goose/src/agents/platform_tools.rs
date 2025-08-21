use indoc::indoc;
use rmcp::model::{Tool, ToolAnnotations};
use rmcp::object;

pub const PLATFORM_READ_RESOURCE_TOOL_NAME: &str = "platform__read_resource";
pub const PLATFORM_LIST_RESOURCES_TOOL_NAME: &str = "platform__list_resources";
pub const PLATFORM_SEARCH_AVAILABLE_EXTENSIONS_TOOL_NAME: &str =
    "platform__search_available_extensions";
pub const PLATFORM_MANAGE_EXTENSIONS_TOOL_NAME: &str = "platform__manage_extensions";
pub const PLATFORM_MANAGE_SCHEDULE_TOOL_NAME: &str = "platform__manage_schedule";
pub const PLATFORM_SEARCH_MESSAGES_TOOL_NAME: &str = "platform__search_messages";
pub const PLATFORM_SEARCH_DOCUMENTS_TOOL_NAME: &str = "platform__search_documents";

pub fn read_resource_tool() -> Tool {
    Tool::new(
        PLATFORM_READ_RESOURCE_TOOL_NAME.to_string(),
        indoc! {r#"
            Read a resource from an extension.

            Resources allow extensions to share data that provide context to LLMs, such as
            files, database schemas, or application-specific information. This tool searches for the
            resource URI in the provided extension, and reads in the resource content. If no extension
            is provided, the tool will search all extensions for the resource.
        "#}.to_string(),
        object!({
            "type": "object",
            "required": ["uri"],
            "properties": {
                "uri": {"type": "string", "description": "Resource URI"},
                "extension_name": {"type": "string", "description": "Optional extension name"}
            }
        })
    ).annotate(ToolAnnotations {
        title: Some("Read a resource".to_string()),
        read_only_hint: Some(true),
        destructive_hint: Some(false),
        idempotent_hint: Some(false),
        open_world_hint: Some(false),
    })
}

pub fn list_resources_tool() -> Tool {
    Tool::new(
        PLATFORM_LIST_RESOURCES_TOOL_NAME.to_string(),
        indoc! {r#"
            List resources from an extension(s).

            Resources allow extensions to share data that provide context to LLMs, such as
            files, database schemas, or application-specific information. This tool lists resources
            in the provided extension, and returns a list for the user to browse. If no extension
            is provided, the tool will search all extensions for the resource.
        "#}
        .to_string(),
        object!({
            "type": "object",
            "properties": {
                "extension_name": {"type": "string", "description": "Optional extension name"}
            }
        }),
    )
    .annotate(ToolAnnotations {
        title: Some("List resources".to_string()),
        read_only_hint: Some(true),
        destructive_hint: Some(false),
        idempotent_hint: Some(false),
        open_world_hint: Some(false),
    })
}

pub fn search_available_extensions_tool() -> Tool {
    Tool::new(
        PLATFORM_SEARCH_AVAILABLE_EXTENSIONS_TOOL_NAME.to_string(),
        "Searches for additional extensions available to help complete tasks.
        Use this tool when you're unable to find a specific feature or functionality you need to complete your task, or when standard approaches aren't working.
        These extensions might provide the exact tools needed to solve your problem.
        If you find a relevant one, consider using your tools to enable it.".to_string(),
        object!({
            "type": "object",
            "required": [],
            "properties": {}
        })
    ).annotate(ToolAnnotations {
        title: Some("Discover extensions".to_string()),
        read_only_hint: Some(true),
        destructive_hint: Some(false),
        idempotent_hint: Some(false),
        open_world_hint: Some(false),
    })
}

pub fn manage_extensions_tool() -> Tool {
    Tool::new(
        PLATFORM_MANAGE_EXTENSIONS_TOOL_NAME.to_string(),
        "Tool to manage extensions and tools in goose context.
            Enable or disable extensions to help complete tasks.
            Enable or disable an extension by providing the extension name.
            "
        .to_string(),
        object!({
            "type": "object",
            "required": ["action", "extension_name"],
            "properties": {
                "action": {"type": "string", "description": "The action to perform", "enum": ["enable", "disable"]},
                "extension_name": {"type": "string", "description": "The name of the extension to enable"}
            }
        }),
    ).annotate(ToolAnnotations {
        title: Some("Enable or disable an extension".to_string()),
        read_only_hint: Some(false),
        destructive_hint: Some(false),
        idempotent_hint: Some(false),
        open_world_hint: Some(false),
    })
}

pub fn manage_schedule_tool() -> Tool {
    Tool::new(
        PLATFORM_MANAGE_SCHEDULE_TOOL_NAME.to_string(),
        indoc! {r#"
            Manage scheduled recipe execution for this Goose instance.
            
            Actions:
            - "list": List all scheduled jobs
            - "create": Create a new scheduled job from a recipe file
            - "run_now": Execute a scheduled job immediately  
            - "pause": Pause a scheduled job
            - "unpause": Resume a paused job
            - "delete": Remove a scheduled job
            - "kill": Terminate a currently running job
            - "inspect": Get details about a running job
            - "sessions": List execution history for a job
            - "session_content": Get the full content (messages) of a specific session
        "#}
        .to_string(),
        object!({
            "type": "object",
            "required": ["action"],
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["list", "create", "run_now", "pause", "unpause", "delete", "kill", "inspect", "sessions", "session_content"]
                },
                "job_id": {"type": "string", "description": "Job identifier for operations on existing jobs"},
                "recipe_path": {"type": "string", "description": "Path to recipe file for create action"},
                "cron_expression": {"type": "string", "description": "A cron expression for create action. Supports both 5-field (minute hour day month weekday) and 6-field (second minute hour day month weekday) formats. 5-field expressions are automatically converted to 6-field by prepending '0' for seconds."},
                "execution_mode": {"type": "string", "description": "Execution mode for create action: 'foreground' or 'background'", "enum": ["foreground", "background"], "default": "background"},
                "limit": {"type": "integer", "description": "Limit for sessions list", "default": 50},
                "session_id": {"type": "string", "description": "Session identifier for session_content action"}
            }
        }),
    ).annotate(ToolAnnotations {
        title: Some("Manage scheduled recipes".to_string()),
        read_only_hint: Some(false),
        destructive_hint: Some(true), // Can kill jobs
        idempotent_hint: Some(false),
        open_world_hint: Some(false),
    })
}

#[cfg(feature = "vectordb-sqlite")]
pub fn search_messages_tool() -> Tool {
    Tool::new(
        PLATFORM_SEARCH_MESSAGES_TOOL_NAME.to_string(),
        indoc! {r#"
            Search through conversation messages using vector similarity.
            
            This tool allows you to find messages from past conversations that are semantically
            similar to your query. It uses vector embeddings to understand the meaning and context
            of messages, not just keyword matching.
            
            Use this when you need to:
            - Find previous discussions about a topic
            - Locate relevant context from past conversations
            - Discover how similar problems were solved before
            - Search for specific information mentioned in conversations
        "#}
        .to_string(),
        object!({
            "type": "object",
            "required": ["query"],
            "properties": {
                "query": {
                    "type": "string", 
                    "description": "The search query - describe what you're looking for in natural language"
                },
                "limit": {
                    "type": "integer", 
                    "description": "Maximum number of similar messages to return (default: 5)",
                    "default": 5,
                    "minimum": 1,
                    "maximum": 20
                },
                "session_filter": {
                    "type": "string", 
                    "description": "Optional session ID to limit search to a specific conversation"
                }
            }
        }),
    ).annotate(ToolAnnotations {
        title: Some("Search conversation messages".to_string()),
        read_only_hint: Some(true),
        destructive_hint: Some(false),
        idempotent_hint: Some(true),
        open_world_hint: Some(false),
    })
}

#[cfg(feature = "vectordb-sqlite")]
pub fn search_documents_tool() -> Tool {
    Tool::new(
        PLATFORM_SEARCH_DOCUMENTS_TOOL_NAME.to_string(),
        indoc! {r#"
            Search through indexed documents using vector similarity.
            
            This tool allows you to find document content that is semantically similar to your query.
            It searches through files that have been indexed into the document vector database,
            including code files, documentation, configuration files, and other text-based content.
            
            Use this when you need to:
            - Find relevant code examples or implementations
            - Locate documentation about specific topics
            - Search for configuration patterns
            - Discover similar functionality across files
            - Find where specific concepts are discussed
        "#}
        .to_string(),
        object!({
            "type": "object",
            "required": ["query"],
            "properties": {
                "query": {
                    "type": "string", 
                    "description": "The search query - describe what you're looking for in natural language"
                },
                "limit": {
                    "type": "integer", 
                    "description": "Maximum number of similar document chunks to return (default: 10)",
                    "default": 10,
                    "minimum": 1,
                    "maximum": 50
                },
                "content_type_filter": {
                    "type": "string", 
                    "description": "Optional filter by content type (e.g., 'rust', 'markdown', 'python', 'javascript', 'json', 'yaml')"
                }
            }
        }),
    ).annotate(ToolAnnotations {
        title: Some("Search indexed documents".to_string()),
        read_only_hint: Some(true),
        destructive_hint: Some(false),
        idempotent_hint: Some(true),
        open_world_hint: Some(false),
    })
}
