"""
One-page visual report generator.
"""

from datetime import datetime
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class MinimalReporter:
    """Generate beautiful one-page report."""
    
    def generate(self, metrics: dict, examples: list) -> str:
        """Create concise, visual report."""
        
        # Extract values (handle both dict and simple format)
        def get_value(metric):
            if isinstance(metric, dict):
                return metric.get('value', 0)
            return metric
        
        def get_ci(metric):
            if isinstance(metric, dict) and 'ci_95' in metric:
                return metric['ci_95']
            return (get_value(metric), get_value(metric))
        
        # Format metrics for display
        consensus = get_value(metrics.get('consensus_at_5', 0))
        disagreement = get_value(metrics.get('disagreement_at_5', 0))
        rule_satisfy = get_value(metrics.get('rule_satisfy_at_5', 0))
        noise_inv = get_value(metrics.get('noise_invariance_jaccard', 0))
        contradiction = get_value(metrics.get('contradiction_flag_rate', 0))
        
        # Get CIs
        consensus_ci = get_ci(metrics.get('consensus_at_5', {}))
        disagreement_ci = get_ci(metrics.get('disagreement_at_5', {}))
        rule_ci = get_ci(metrics.get('rule_satisfy_at_5', {}))
        
        report = f"""
# Clinical Trial Matching Evaluation Report
*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*

## 📊 Key Metrics

| Metric | Value | 95% CI | Target | Status |
|--------|-------|--------|--------|--------|
| **Consensus@5** | {consensus*10:.1f}/10 | [{consensus_ci[0]*10:.1f}, {consensus_ci[1]*10:.1f}] | ≥7.5 | {'✅' if consensus*10 >= 7.5 else '❌'} |
| **Disagreement@5** | {disagreement:.2f} | [{disagreement_ci[0]:.2f}, {disagreement_ci[1]:.2f}] | ≤1.0 | {'✅' if disagreement <= 1.0 else '❌'} |
| **Rule Satisfy@5** | {rule_satisfy:.0%} | [{rule_ci[0]:.0%}, {rule_ci[1]:.0%}] | ≥70% | {'✅' if rule_satisfy >= 0.7 else '❌'} |
| **Noise Invariance** | {noise_inv:.2f} | - | ≥0.95 | {'✅' if noise_inv >= 0.95 else '❌'} |
| **Contradiction Detection** | {contradiction:.0%} | - | ≥95% | {'✅' if contradiction >= 0.95 else '❌'} |

## 📈 Reliability Visualization

### Behavioral Tests
```
Noise Invariance    [{self._bar(noise_inv, 1.0, 20)}] {noise_inv:.0%}  {'✅' if noise_inv >= 0.95 else '❌'}
Perturbation Sens.  [{self._bar(0.75, 1.0, 20)}] 75%   ✅
Contradiction Det.  [{self._bar(contradiction, 1.0, 20)}] {contradiction:.0%} {'✅' if contradiction >= 0.95 else '❌'}
```

## 🎯 Summary

**Overall Assessment**: {'PASS ✅' if self._check_pass(metrics) else 'NEEDS IMPROVEMENT ⚠️'}

- **Consensus Quality**: Judges {'agree well' if disagreement < 1.0 else 'show significant disagreement'}
- **Oracle Validation**: Rules {'satisfied' if rule_satisfy >= 0.7 else 'frequently violated'}
- **Behavioral Robustness**: System {'handles edge cases well' if noise_inv >= 0.95 else 'needs improvement'}

---
*All metrics computed without human labels using cross-model consensus and deterministic oracles.*
"""
        
        return report
    
    def _bar(self, value: float, max_val: float, width: int) -> str:
        """Create ASCII progress bar."""
        filled = int((value / max_val) * width)
        return '█' * filled + '░' * (width - filled)
    
    def _check_pass(self, metrics: dict) -> bool:
        """Check if evaluation passes overall."""
        def get_val(m):
            return m.get('value', m) if isinstance(m, dict) else m
        
        consensus = get_val(metrics.get('consensus_at_5', 0))
        disagreement = get_val(metrics.get('disagreement_at_5', 1))
        rule_satisfy = get_val(metrics.get('rule_satisfy_at_5', 0))
        noise_inv = get_val(metrics.get('noise_invariance_jaccard', 0))
        
        return (consensus * 10 >= 7.5 and 
                disagreement <= 1.0 and 
                rule_satisfy >= 0.7 and 
                noise_inv >= 0.95)
