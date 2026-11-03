import type { SchemaType } from '../../schema';

export function getBaseType(type?: SchemaType): string {
  if (!type) return 'string';
  const t = type as unknown as Record<string, unknown>;
  if (typeof t.type === 'string') return t.type;
  if (Array.isArray(t.anyOf)) {
    for (const variant of t.anyOf as Record<string, unknown>[]) {
      if (typeof variant.type === 'string' && variant.type !== 'null') {
        return variant.type;
      }
    }
  }
  return 'string';
}

export function isNullable(type?: SchemaType): boolean {
  if (!type) return false;
  const t = type as unknown as Record<string, unknown>;
  if (Array.isArray(t.anyOf)) {
    return (t.anyOf as Record<string, unknown>[]).some(v => v.type === 'null');
  }
  return false;
}

export function getEnumValues(type?: SchemaType): (string | number)[] | undefined {
  if (!type) return undefined;
  const t = type as unknown as Record<string, unknown>;
  if (Array.isArray(t.enum)) return t.enum as (string | number)[];
  if (Array.isArray(t.anyOf)) {
    for (const variant of t.anyOf as Record<string, unknown>[]) {
      if (Array.isArray(variant.enum)) return variant.enum as (string | number)[];
    }
  }
  return undefined;
}

export function isTensorOrInitValue(value: unknown): boolean {
  if (value && typeof value === 'object' && !Array.isArray(value)) {
    const obj = value as Record<string, unknown>;
    return '$tensor' in obj || '$init' in obj || '$fn' in obj;
  }
  return false;
}
