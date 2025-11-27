use std::path::Path;

use rsgrep::{
   chunker::{
      Chunker, anchor::create_anchor_chunk, create_chunker, fallback::FallbackChunker,
      treesitter::TreeSitterChunker,
   },
   types::ChunkType,
};

#[test]
fn test_fallback_chunker() {
   let chunker = FallbackChunker::new();
   let content = "line 1\nline 2\nline 3\n";
   let path = Path::new("test.txt");

   let chunks = chunker.chunk(content, path).unwrap();
   assert!(!chunks.is_empty());
   assert_eq!(chunks[0].chunk_type, ChunkType::Block);
}

#[test]
fn test_create_anchor_chunk() {
   let content = r#"
// This is a comment
import { foo } from 'bar';
export const baz = 42;

function test() {
  return true;
}
"#;
   let path = Path::new("test.ts");
   let chunk = create_anchor_chunk(content, path);

   assert!(chunk.is_anchor);
   assert_eq!(chunk.chunk_type, ChunkType::Block);
   assert!(chunk.content.contains("Imports:"));
   assert!(chunk.content.contains("Exports:"));
}

#[test]
fn test_chunker_factory() {
   let ts_chunker = create_chunker(Path::new("test.ts"));
   let txt_chunker = create_chunker(Path::new("test.txt"));

   assert!(std::any::type_name_of_val(&*ts_chunker).contains("TreeSitterChunker"));
   assert!(std::any::type_name_of_val(&*txt_chunker).contains("FallbackChunker"));
}

#[test]
fn test_treesitter_chunker_typescript() {
   let chunker = TreeSitterChunker::new();
   let content = r#"
export function greet(name: string): string {
  return `Hello, ${name}`;
}

export class Person {
  constructor(private name: string) {}

  getName(): string {
    return this.name;
  }
}
"#;
   let path = Path::new("test.ts");

   let result = chunker.chunk(content, path);
   assert!(result.is_ok());
   let chunks = result.unwrap();

   assert!(!chunks.is_empty());
   let has_function = chunks.iter().any(|c| c.chunk_type == ChunkType::Function);
   let has_class = chunks.iter().any(|c| c.chunk_type == ChunkType::Class);

   assert!(has_function || has_class);
}
