import { createClient } from '@supabase/supabase-js'

// You will need to set these in your .env.local
const supabase = createClient(process.env.NEXT_PUBLIC_SUPABASE_URL!, process.env.SUPABASE_SERVICE_ROLE_KEY!)

async function seed() {
  console.log('Seeding data...')

  // 1. Create a Fake User (You should ideally map this to your real Auth ID later)
  // For seeding, we just create the content.
  
  // 2. Key Results
  const { data: kr, error: krError } = await supabase.from('key_results').insert([
    { title: 'Find Job Outside India', deadline: '2025-04-01', status: 'active' },
    { title: 'Generate Alternate Income', deadline: '2025-06-01', status: 'active' }
  ]).select()

  if (krError) console.error(krError)
  if (!kr) return

  const jobKR = kr[0]
  const incomeKR = kr[1]

  // 3. Hypotheses
  const { data: hypos } = await supabase.from('hypotheses').insert([
    // Job KR Hypotheses
    { 
      kr_id: jobKR.id, 
      title: 'Aggressive LinkedIn Outreach', 
      confidence_score: 0.65, 
      status: 'active' 
    },
    { 
      kr_id: jobKR.id, 
      title: 'Vibe Coder Personal Brand', 
      confidence_score: 0.40, 
      status: 'active' 
    },
    // Income KR Hypotheses
    { 
      kr_id: incomeKR.id, 
      title: 'Cloud Kitchen (Failed)', 
      confidence_score: 0.1, 
      status: 'graveyard',
      graveyard_entry_date: new Date().toISOString(),
      consolidated_learnings: 'Margins were too low due to licensing fees.'
    }
  ]).select()

  // 4. Experiments
  if (hypos) {
    const linkedInHypo = hypos.find(h => h.title.includes('LinkedIn'))
    if (linkedInHypo) {
      await supabase.from('experiments').insert([
        { hypothesis_id: linkedInHypo.id, title: 'Connect with 5 alumni', status: 'todo' },
        { hypothesis_id: linkedInHypo.id, title: 'Draft cold email template', status: 'done', learning_note: 'Short templates work better.' }
      ])
    }
  }

  console.log('Seeding complete.')
}

seed()
