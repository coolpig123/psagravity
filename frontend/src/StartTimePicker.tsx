import { useState } from 'react'

// Start time of the prediction. The value is always a UTC ISO string like 2026-08-09T21:00Z (what the server gets);
// the picker shows it in the browser's local time (calendar + clock), or as typed UTC text with the UTC toggle.

const HOUR_MS = 3_600_000
const two = (n: number) => String(n).padStart(2, '0')

/** Parse the UTC value; null when it is not a valid time (e.g. half typed). */
function parse(value: string): Date | null {
  const t = value.trim()
  if (!t) return null
  const d = new Date(t)
  return Number.isNaN(d.getTime()) ? null : d
}

/** Date -> '2026-08-09T21:00Z' (minutes precision, same format as before). */
function toUtcIso(d: Date): string {
  return `${d.getUTCFullYear()}-${two(d.getUTCMonth() + 1)}-${two(d.getUTCDate())}T${two(d.getUTCHours())}:${two(d.getUTCMinutes())}Z`
}

/** Date -> value of <input type="datetime-local"> in the browser's time zone. */
function toLocalInput(d: Date): string {
  return `${d.getFullYear()}-${two(d.getMonth() + 1)}-${two(d.getDate())}T${two(d.getHours())}:${two(d.getMinutes())}`
}

/** The current time rounded down to the full hour. */
function nowHour(): Date {
  return new Date(Math.floor(Date.now() / HOUR_MS) * HOUR_MS)
}

export default function StartTimePicker({ value, onChange }: { value: string; onChange: (utcIso: string) => void }) {
  const [typeUtc, setTypeUtc] = useState(false) // false: local calendar + clock, true: type the UTC value as text
  const d = parse(value)
  const zone = Intl.DateTimeFormat().resolvedOptions().timeZone
  const nudge = (hours: number) => d && onChange(toUtcIso(new Date(d.getTime() + hours * HOUR_MS)))

  return (
    <div className="start start-time">
      <span>Start</span>
      <div className="start-field">
        {typeUtc ? (
          <input value={value} onChange={(e) => onChange(e.target.value)} placeholder="2026-08-09T21:00Z" size={18}
            aria-label="Start time (UTC)" />
        ) : (
          <input
            type="datetime-local"
            value={d ? toLocalInput(d) : ''}
            // datetime-local is read in the browser's time zone: new Date('YYYY-MM-DDTHH:MM') is local time
            onChange={(e) => e.target.value && onChange(toUtcIso(new Date(e.target.value)))}
            aria-label={`Start time (local, ${zone})`}
            title={`Your local time (${zone})`}
          />
        )}
        <small className="start-other">
          {!d ? 'not a valid time' : typeUtc ? `= ${toLocalInput(d).replace('T', ' ')} local` : `= ${value.replace('T', ' ').replace('Z', '')} UTC`}
        </small>
      </div>
      <button onClick={() => setTypeUtc(!typeUtc)} title={typeUtc ? 'Pick in your local time' : 'Type the time in UTC'}>
        {typeUtc ? 'Local' : 'UTC'}
      </button>
      <button onClick={() => nudge(-1)} disabled={!d} title="One hour earlier">−1h</button>
      <button onClick={() => nudge(1)} disabled={!d} title="One hour later">+1h</button>
      <button onClick={() => onChange(toUtcIso(nowHour()))} title="Now, rounded down to the full hour">Now</button>
    </div>
  )
}
