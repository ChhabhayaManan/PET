import { useState } from 'react'
import { useAuth } from '../../../hooks/useAuth'
import { TextField } from '../../../components/ui/Field'
import { AboutFields, aboutForm, aboutPayload, aboutComplete } from '../../../components/onboarding/BackgroundFields'
import { StepTitle, StepNav, useSaveStep } from '../../../components/onboarding/StepControls'

export default function AboutStep({ initial, onSaved, onBack, onConflict }) {
  const { user, updateUser } = useAuth()
  const [name, setName] = useState(() => initial?.full_name ?? user?.full_name ?? '')
  const [form, setForm] = useState(() => aboutForm(initial, user))
  const handleSaved = (state) => {
    updateUser({ full_name: state.answers?.about?.full_name ?? name.trim() })
    onSaved(state)
  }
  const { saving, save } = useSaveStep('about', handleSaved, onConflict)

  return (
    <div>
      <StepTitle eyebrow="About you" title="A little about you" subtitle="This helps PET pitch questions and explanations at the right level." />
      <TextField label="Your name" maxLength={255} value={name} autoComplete="name"
        onChange={e => setName(e.target.value)} placeholder="What should PET call you?" className="mb-6" />
      <AboutFields form={form} onChange={setForm} />
      <StepNav onBack={onBack} onNext={() => save({ ...aboutPayload(form), full_name: name.trim() })} saving={saving}
        canNext={!!name.trim() && aboutComplete(form)} />
    </div>
  )
}
