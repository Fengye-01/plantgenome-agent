import json

with open('docs/evaluation_results.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print('=== literature_search 用例详情 ===')
for item in data['details']:
    if item['category'] == 'literature_search':
        print(f"[{item['id']:2d}] {item['question'][:50]}...")
        print(f"  intent: {item['intent']}")
        print(f"  tool: {item['tool_name']}")
        print(f"  tool_correct: {item['tool_correct']}")
        print(f"  hit_at_3: {item['hit_at_3']}")
        print(f"  expected_source: {item['expected_source']}")
        print(f"  sources_count: {item['sources_count']}")
        print(f"  keyword_hits: {item['keyword_hits']}")
        print(f"  answer_preview: {item['answer_preview'][:80]}...")
        print()
