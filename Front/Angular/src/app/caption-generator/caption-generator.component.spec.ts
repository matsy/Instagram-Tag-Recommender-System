import { ComponentFixture, TestBed } from '@angular/core/testing';

import { CaptionGeneratorComponent } from './caption-generator.component';

describe('CaptionGeneratorComponent', () => {
  let component: CaptionGeneratorComponent;
  let fixture: ComponentFixture<CaptionGeneratorComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ CaptionGeneratorComponent ]
    })
    .compileComponents();
  });

  beforeEach(() => {
    fixture = TestBed.createComponent(CaptionGeneratorComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
